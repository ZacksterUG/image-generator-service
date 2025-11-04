from pathlib import Path
from typing import Dict, Any

from model.src.helpers import MessageComparer
from model.src.model.builder_model import ModelType, ModelFactory
from model.src.model.concrete_models.butterfly_model import ButterflyModel
from model.src.model.concrete_models.cat_model import CatModel
from model.src.model.facade_model import FacadeModel
from model.src.queue.queue_base import QueueBase
from model.src.queue.rabbit_mq.rabbit_mq_queue import RabbitMQQueue


def init_models(cfg: Dict[str, Any]):
    base_path = Path(cfg['MODELS_BASE_DIR'])
    cat_model_path = base_path / cfg['CAT_MODEL_WEIGHTS_FILENAME']
    butter_fly_path = base_path / cfg['BUTTERFLY_MODEL_WEIGHTS_FILENAME']

    cat_model = ModelFactory.create_model(ModelType.CAT, cat_model_path, cfg['DEVICE'])
    butterfly_mode = ModelFactory.create_model(ModelType.BUTTERFLY, butter_fly_path, cfg['DEVICE'])

    # Проверяем, что это действительно CatModel
    if isinstance(cat_model, CatModel):
        cat_model.set_timestamps(cfg['CAT_MODEL_TIMESTAMPS'])

    if isinstance(butterfly_mode, ButterflyModel):
        butterfly_mode.set_timestamps(cfg['BUTTERFLY_MODEL_TIMESTAMPS'])

    facade = FacadeModel(cat_model, butterfly_mode)

    return facade


def init_comparer():
    comparer = MessageComparer()
    return comparer

def init_queue(
    queue_name: str,
    *,
    durable: bool = True,
    prefetch_count: int = 1,
    cfg: Dict[str, Any],
) -> QueueBase:
    """Инициализирует и возвращает обработчик очереди RabbitMQ.

    Параметры подключения загружаются из .env файла (если он существует)
    или из переменных окружения.

    Args:
        queue_name (str): Имя очереди, с которой будет работать экземпляр.
        durable (bool): Должна ли очередь быть устойчивой к перезапуску брокера.
        prefetch_count (int): Макс. число неподтверждённых сообщений на consumer.
        cfg (Dict[str, Any]): Конфиги подключения к RabbitMQ сервера.

    Returns:
        QueueBase: Экземпляр обработчика очереди, реализующий интерфейс QueueBase.
    """


    # Читаем параметры из окружения (с fallback на localhost/guest)
    host = cfg["RABBITMQ_HOST"]
    port = cfg["RABBITMQ_PORT"]
    username = cfg["RABBITMQ_USER"]
    password = cfg["RABBITMQ_PASSWORD"]
    virtual_host = cfg["RABBITMQ_VIRTUAL_HOST"]

    return RabbitMQQueue(
        host=host,
        port=port,
        username=username,
        password=password,
        virtual_host=virtual_host,
        queue_name=queue_name,
        durable=durable,
        prefetch_count=prefetch_count,
    )