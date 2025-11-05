import json
import logging
from typing import Any, Optional
from abc import ABC
from pika import (
    BlockingConnection,
    ConnectionParameters,
    PlainCredentials,
    BasicProperties,
    exceptions as pika_exceptions
)
from pika.adapters.blocking_connection import BlockingChannel

from model.src.queue.queue_base import QueueBase, Message


logger = logging.getLogger(__name__)


class RabbitMQQueue(QueueBase):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/",
        queue_name: str = "default",  # очередь, с которой работает экземпляр
        durable: bool = True,
        prefetch_count: int = 1,
    ):
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._virtual_host = virtual_host
        self._queue_name = queue_name
        self._durable = durable
        self._prefetch_count = prefetch_count

        self._connection: Optional[BlockingConnection] = None
        self._channel: Optional[BlockingChannel] = None
        self._needs_reinit = True

        self._connect()
        self._declare_queue()
        self._needs_reinit = False

    def _ensure_channel(self):
        """Гарантирует, что соединение и канал открыты. Переподключается при необходимости."""
        # Проверяем соединение
        if not self._connection or self._connection.is_closed:
            logger.debug("RabbitMQ connection is closed or missing. Reconnecting...")
            self._connect()
            self._declare_queue()
            self._needs_reinit = False
            return

        # Проверяем канал
        # Если флаг установлен или канал закрыт, инициализируем заново
        if self._needs_reinit or not self._channel or self._channel.is_closed:
            logger.debug("RabbitMQ channel needs reinitialization or is closed. Reinitializing channel...")
            try:
                # Если канал открыт, закрываем его аккуратно перед пересозданием
                if self._channel and self._channel.is_open:
                    self._channel.close()
            except pika_exceptions.ChannelClosedByBroker as e:
                 logger.debug(f"Channel already closed by broker during reinit: {e}")
            except Exception as e:
                 logger.debug(f"Error closing old channel during reinit: {e}")
            # Создаем новый канал
            self._channel = self._connection.channel()
            self._channel.basic_qos(prefetch_count=self._prefetch_count)
            self._declare_queue()
            self._needs_reinit = False

    def _connect(self):
        """Устанавливает соединение и канал с RabbitMQ."""
        credentials = PlainCredentials(self._username, self._password)
        params = ConnectionParameters(
            host=self._host,
            port=self._port,
            virtual_host=self._virtual_host,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
            # Добавим параметры для лучшего поведения при разрывах
            socket_timeout=10,
            # connection_attempts=3, # Количество попыток подключения при старте
            # retry_delay=2,        # Задержка между попытками
        )
        self._connection = BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.basic_qos(prefetch_count=self._prefetch_count)
        # Сбросим флаг при новом подключении
        self._needs_reinit = False


    def _declare_queue(self):
        """Объявляет очередь с заданными параметрами."""
        if self._channel:
            self._channel.queue_declare(
                queue=self._queue_name,
                durable=self._durable,
                exclusive=False,
                auto_delete=False,
            )

    def declare_queue(self, queue: str) -> bool:
        """Объявляет новую очередь (редко используется, если экземпляр привязан к одной очереди)."""
        try:
            self._ensure_channel() # Убедиться, что канал жив
            self._channel.queue_declare(
                queue=queue,
                durable=self._durable,
                exclusive=False,
                auto_delete=False,
            )
            return True
        except pika_exceptions.AMQPError as e: # Используем более конкретный тип
            logger.error(f"Failed to declare queue '{queue}': {e}")
            # Помечаем, что нужно переподключиться при следующем вызове
            self._needs_reinit = True
            return False
        except Exception as e:
            logger.error(f"Unexpected error during queue declaration: {e}")
            self._needs_reinit = True
            return False

    def push(self, queue: str, data: Any) -> bool:
        """Отправляет сообщение в указанную очередь."""
        try:
            # Обернем вызов в try-except и повторим один раз при сбое соединения
            # Это помогает избежать сбоя с первого раза
            return self._push_internal(queue, data)
        except (pika_exceptions.AMQPError, pika_exceptions.StreamLostError, pika_exceptions.ConnectionClosedByBroker, pika_exceptions.ChannelClosedByBroker) as e:
             logger.warning(f"Initial push failed due to connection issue: {e}. Retrying...")
             try:
                 self._ensure_channel() # Принудительно переподключиться
                 return self._push_internal(queue, data)
             except pika_exceptions.AMQPError as retry_e:
                 logger.error(f"Failed to push to queue '{queue}' after retry: {retry_e}")
                 self._needs_reinit = True # Пометить для переподключения в следующий раз
                 return False
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize message: {e}")
            return False

    def _push_internal(self, queue: str, data: Any) -> bool:
        """Внутренняя функция отправки, вызываемая из push."""
        self._ensure_channel() # Убедиться, что канал жив перед отправкой
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=body,
            properties=BasicProperties(
                delivery_mode=2,  # make message persistent
            ),
        )
        return True

    def pop(self) -> Optional[Message]:
        """Извлекает одно сообщение без подтверждения."""
        try:
            # Обернем вызов в try-except и повторим один раз при сбое соединения
            return self._pop_internal()
        except (pika_exceptions.AMQPError, pika_exceptions.StreamLostError, pika_exceptions.ConnectionClosedByBroker, pika_exceptions.ChannelClosedByBroker) as e:
             logger.warning(f"Initial pop failed due to connection issue: {e}. Retrying...")
             try:
                 self._ensure_channel() # Принудительно переподключиться
                 return self._pop_internal()
             except pika_exceptions.AMQPError as retry_e:
                 logger.error(f"Failed to pop from queue '{self._queue_name}' after retry: {retry_e}")
                 self._needs_reinit = True # Пометить для переподключения в следующий раз
                 return None
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to decode message: {e}")
            return None

    def _pop_internal(self) -> Optional[Message]:
        """Внутренняя функция получения, вызываемая из pop."""
        self._ensure_channel() # Убедиться, что канал жив перед получением
        method_frame, header_frame, body = self._channel.basic_get(
            queue=self._queue_name,
            auto_ack=False  # важно: не подтверждать автоматически
        )
        if method_frame is None:
            return None  # очередь пуста

        delivery_tag = method_frame.delivery_tag
        decoded_body = json.loads(body.decode("utf-8"))
        return Message(body=decoded_body, delivery_tag=delivery_tag)

    def ack(self, delivery_tag: Any) -> bool:
        """Подтверждает обработку сообщения."""
        try:
            return self._ack_internal(delivery_tag)
        except (pika_exceptions.AMQPError, pika_exceptions.StreamLostError, pika_exceptions.ConnectionClosedByBroker, pika_exceptions.ChannelClosedByBroker) as e:
             logger.warning(f"Initial ack failed due to connection issue: {e}. Retrying...")
             try:
                 self._ensure_channel() # Принудительно переподключиться
                 return self._ack_internal(delivery_tag)
             except pika_exceptions.AMQPError as retry_e:
                 logger.error(f"Failed to ack message {delivery_tag} after retry: {retry_e}")
                 self._needs_reinit = True # Пометить для переподключения в следующий раз
                 return False

    def _ack_internal(self, delivery_tag: Any) -> bool:
        """Внутренняя функция подтверждения, вызываемая из ack."""
        self._ensure_channel() # Убедиться, что канал жив перед подтверждением
        self._channel.basic_ack(delivery_tag=delivery_tag)
        return True

    def nack(self, delivery_tag: Any, requeue: bool = True) -> bool:
        """Отклоняет сообщение."""
        try:
            return self._nack_internal(delivery_tag, requeue)
        except (pika_exceptions.AMQPError, pika_exceptions.StreamLostError, pika_exceptions.ConnectionClosedByBroker, pika_exceptions.ChannelClosedByBroker) as e:
             logger.warning(f"Initial nack failed due to connection issue: {e}. Retrying...")
             try:
                 self._ensure_channel() # Принудительно переподключиться
                 return self._nack_internal(delivery_tag, requeue)
             except pika_exceptions.AMQPError as retry_e:
                 logger.error(f"Failed to nack message {delivery_tag} after retry: {retry_e}")
                 self._needs_reinit = True # Пометить для переподключения в следующий раз
                 return False

    def _nack_internal(self, delivery_tag: Any, requeue: bool = True) -> bool:
        """Внутренняя функция отклонения, вызываемая из nack."""
        self._ensure_channel() # Убедиться, что канал жив перед отклонением
        self._channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
        return True

    def empty(self) -> bool:
        """Проверяет, пуста ли очередь (приблизительно)."""
        try:
            # Обернем вызов в try-except и повторим один раз при сбое соединения
            return self._empty_internal()
        except (pika_exceptions.AMQPError, pika_exceptions.StreamLostError, pika_exceptions.ConnectionClosedByBroker, pika_exceptions.ChannelClosedByBroker) as e:
             logger.warning(f"Initial empty check failed due to connection issue: {e}. Retrying...")
             try:
                 self._ensure_channel() # Принудительно переподключиться
                 return self._empty_internal()
             except pika_exceptions.AMQPError as retry_e:
                 logger.error(f"Failed to check empty status for queue '{self._queue_name}' after retry: {retry_e}")
                 self._needs_reinit = True # Пометить для переподключения в следующий раз
                 return True  # считаем пустой при ошибке
        except Exception:
            return True  # считаем пустой при ошибке

    def _empty_internal(self) -> bool:
        """Внутренняя функция проверки пустоты, вызываемая из empty."""
        self._ensure_channel() # Убедиться, что канал жив перед проверкой
        method = self._channel.queue_declare(
            queue=self._queue_name, passive=True
        )
        return method.method.message_count == 0

    def close(self) -> None:
        """Закрывает соединение."""
        try:
            if self._channel and self._channel.is_open:
                self._channel.close()
        except Exception as e:
            logger.warning(f"Error closing channel: {e}")
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    def ping(self) -> dict:
        """Проверяет работоспособность соединения с RabbitMQ.

        Returns:
            dict: {"error": bool, "message": str}
        """
        try:
            # Используем внутреннюю функцию с повторной попыткой
            self._ensure_channel()
            self._channel.queue_declare(queue=self._queue_name, passive=True)
            return {"error": False, "message": "OK"}
        except (pika_exceptions.AMQPError, pika_exceptions.StreamLostError, pika_exceptions.ConnectionClosedByBroker, pika_exceptions.ChannelClosedByBroker) as e:
             logger.warning(f"Ping failed due to connection issue: {e}. Retrying...")
             try:
                 self._ensure_channel() # Принудительно переподключиться
                 self._channel.queue_declare(queue=self._queue_name, passive=True)
                 return {"error": False, "message": "OK"}
             except Exception as retry_e:
                 error_msg = f"RabbitMQ ping failed after retry: {str(retry_e)}"
                 logger.error(error_msg)
                 self._needs_reinit = True # Пометить для переподключения в следующий раз
                 return {"error": True, "message": error_msg}
        except Exception as e:
            error_msg = f"RabbitMQ ping failed: {str(e)}"
            logger.error(error_msg)
            return {"error": True, "message": error_msg}
