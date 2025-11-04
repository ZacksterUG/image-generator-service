import logging
import os
import time
from typing import Callable, Any, List

import numpy as np

from model.config import get_config, init_env
from model.init import init_models, init_comparer, init_queue
from model.src.helpers import MessageComparer, messages_similarity
from model.src.model.facade_model import FacadeModel
from model.src.queue.queue_base import QueueBase
import base64

logger = logging.getLogger(__name__)

def message_compare_carrier(cmp: MessageComparer, eps: float):
    def compare(a: str, b: str):
        enc1 = cmp.encode(a)
        enc2 = cmp.encode(b)

        res = cmp(enc1, enc2)

        return {
            'is_similar': messages_similarity(cmp, a, b, eps),
            'metric': res
        }
    return compare

def validate_body(body: Any) -> dict:
    msgs = []

    _user_id = body.get("user_id")

    if not _user_id:
        msgs.append("Поле 'user_id' отсутствует в сообщении")

    _message = body.get("message")

    if not _message:
        msgs.append("Поле 'message' отсутствует в сообщении")

    return {
        'error': len(msgs) > 0,
        'messages': '\n'.join(msgs),
    }

def get_relevant_class(classes: List[str], message: str, comparer):
    max_value = 1.0
    _relevant_class = None

    for cl in classes:
        res = comparer(cl, message)

        if res['is_similar'] and res['metric'] < max_value:
            max_value = res['metric']
            _relevant_class = cl

    return _relevant_class

def process_queue(
        receiver_queue: QueueBase,
        uploader_queue: QueueBase,
        model_facade: FacadeModel,
        comparer: Callable):
    """Обрабатывает очередь сообщений: получает, генерирует изображение, отправляет результат.

    Args:
        receiver_queue (QueueBase): Очередь входящих запросов (например, "image_generation_requests").
        uploader_queue (QueueBase): Очередь для отправки результатов (например, "telegram_responses").
    """


    while True:
        try:
            msg = receiver_queue.pop()
            if msg is None:
                time.sleep(0.5)
                continue

            logger.info(f"📥 Получено сообщение: {msg.body}")

            try:
                validation_result = validate_body(msg.body)

                if validation_result['error']:
                    raise ValueError(validation_result['messages'])

                user_id = msg.body.get("user_id")
                message = msg.body.get("message")

                relevant_class = get_relevant_class(['cat', 'butterfly'], message, comparer)

                if relevant_class is None:
                    result_payload = {
                        "user_id": user_id,
                        "image_b64": None,
                        "shape": None,
                        "error": True,
                        "message": 'Unknown received class'
                    }

                    success = uploader_queue.push("message_uploader", result_payload)

                    if not success:
                        raise RuntimeError("Не удалось отправить результат в очередь ответов")

                    # Подтверждаем обработку → сообщение удаляется из очереди
                    receiver_queue.ack(msg.delivery_tag)
                    logger.info(f"🟧 Сообщение для user_id={user_id} доставлено с замечанием {result_payload['message']}")

                    continue

                noise = np.random.randn(1, 3, 64, 64).astype(np.float32)
                image_array = model_facade.generate_by_class(relevant_class, noise)[0]

                # Сериализуем ndarray (пример для 3x64x64)
                image_bytes = image_array.tobytes()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')

                result_payload = {
                    "user_id": user_id,
                    "image_b64": image_b64,
                    "shape": list(image_array.shape),
                    "error": False,
                    "message": 'ok'
                }

                # Отправляем в очередь ответов
                success = uploader_queue.push("message_uploader", result_payload)

                if not success:
                    raise RuntimeError("Не удалось отправить результат в очередь ответов")

                # Подтверждаем обработку → сообщение удаляется из очереди
                receiver_queue.ack(msg.delivery_tag)
                logger.info(f"✅ Сообщение для user_id={user_id} успешно обработано")

            except Exception as e:
                logger.exception(f"❌ Ошибка при обработке сообщения: {e}")
                # Отклоняем сообщение. requeue=False — чтобы не зациклиться на "ядовитом" сообщении.
                receiver_queue.nack(msg.delivery_tag, requeue=False)

        except KeyboardInterrupt:
            logger.info("🛑 Получен сигнал завершения. Завершаем обработку...")
            break
        except Exception as e:
            logger.exception(f"⚠️ Критическая ошибка в основном цикле: {e}")
            time.sleep(5)  # пауза перед повтором

def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        print('Инициализация конфигов...')
        init_env()
        cfg = get_config()

        print('Загрузка моделей...')
        model_facade = init_models(cfg)
        message_comparer = init_comparer()

        print('Подключение к серверу очередей...')
        receiver_queuer = init_queue('message_receiver', cfg=cfg)
        uploader_queuer = init_queue('message_uploader', cfg=cfg)
        pong = receiver_queuer.ping()

        if pong['error']:
            raise Exception(pong['message'])

        pong = uploader_queuer.ping()

        if pong['error']:
            raise Exception(pong['message'])

        comparator = message_compare_carrier(message_comparer, cfg['MAX_MESSAGES_DISTANCE'])
    except Exception as e:
        print("Ошибка при инициализации:")
        print(e)

        return

    print('Приложение успешно иницализация, запущена обработка очереди')
    process_queue(receiver_queuer, uploader_queuer, model_facade, comparator)

if __name__ == '__main__':
    main()