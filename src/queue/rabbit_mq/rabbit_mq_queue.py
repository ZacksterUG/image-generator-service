import json
import logging
from typing import Any, Optional
from abc import ABC
from pika import (
    BlockingConnection,
    ConnectionParameters,
    PlainCredentials,
    BasicProperties,
    spec
)
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPError

from model.src.queue.queue_base import QueueBase, Message


logger = logging.getLogger(__name__)


class RabbitMQQueue(QueueBase):
    """Реализация QueueBase для RabbitMQ с поддержкой надёжной обработки."""

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

        self._connect()
        self._declare_queue()

    def _ensure_channel(self):
        """Гарантирует, что соединение и канал открыты. Переподключается при необходимости."""
        try:
            # Проверяем соединение
            if not self._connection or self._connection.is_closed:
                logger.debug("RabbitMQ connection is closed or missing. Reconnecting...")
                self._connect()
                self._declare_queue()
                return

            # Проверяем канал
            if not self._channel or self._channel.is_closed:
                logger.debug("RabbitMQ channel is closed or missing. Reinitializing channel...")
                self._channel = self._connection.channel()
                self._channel.basic_qos(prefetch_count=self._prefetch_count)
                self._declare_queue()

        except Exception as e:
            logger.error(f"Failed to ensure RabbitMQ channel: {e}")
            raise

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
        )
        self._connection = BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.basic_qos(prefetch_count=self._prefetch_count)

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
            self._ensure_channel()
            self._channel.queue_declare(
                queue=queue,
                durable=self._durable,
                exclusive=False,
                auto_delete=False,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to declare queue '{queue}': {e}")
            return False

    def push(self, queue: str, data: Any) -> bool:
        """Отправляет сообщение в указанную очередь."""
        try:
            self._ensure_channel()
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
        except AMQPError as e:
            logger.error(f"Failed to push to queue '{queue}': {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize message: {e}")
            return False

    def pop(self) -> Optional[Message]:
        """Извлекает одно сообщение без подтверждения."""
        try:
            self._ensure_channel()
            method_frame, header_frame, body = self._channel.basic_get(
                queue=self._queue_name,
                auto_ack=False  # важно: не подтверждать автоматически
            )
            if method_frame is None:
                return None  # очередь пуста

            delivery_tag = method_frame.delivery_tag
            decoded_body = json.loads(body.decode("utf-8"))
            return Message(body=decoded_body, delivery_tag=delivery_tag)

        except AMQPError as e:
            logger.error(f"Failed to pop from queue '{self._queue_name}': {e}")
            return None
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to decode message: {e}")
            return None

    def ack(self, delivery_tag: Any) -> bool:
        """Подтверждает обработку сообщения."""
        try:
            self._ensure_channel()
            self._channel.basic_ack(delivery_tag=delivery_tag)
            return True
        except AMQPError as e:
            logger.error(f"Failed to ack message {delivery_tag}: {e}")
            return False

    def nack(self, delivery_tag: Any, requeue: bool = True) -> bool:
        """Отклоняет сообщение."""
        try:
            self._ensure_channel()
            self._channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
            return True
        except AMQPError as e:
            logger.error(f"Failed to nack message {delivery_tag}: {e}")
            return False

    def empty(self) -> bool:
        """Проверяет, пуста ли очередь (приблизительно)."""
        try:
            self._ensure_channel()
            method = self._channel.queue_declare(
                queue=self._queue_name, passive=True
            )
            return method.method.message_count == 0
        except Exception:
            return True  # считаем пустой при ошибке

    def close(self) -> None:
        """Закрывает соединение."""
        if self._channel and self._channel.is_open:
            self._channel.close()
        if self._connection and self._connection.is_open:
            self._connection.close()

    def ping(self) -> dict:
        """Проверяет работоспособность соединения с RabbitMQ.

        Returns:
            dict: {"error": bool, "message": str}
        """
        try:
            self._ensure_channel()
            self._channel.queue_declare(queue=self._queue_name, passive=True)
            return {"error": False, "message": "OK"}
        except Exception as e:
            error_msg = f"RabbitMQ ping failed: {str(e)}"
            logger.error(error_msg)
            return {"error": True, "message": error_msg}

        except Exception as e:
            error_msg = f"RabbitMQ ping failed: {str(e)}"
            logger.error(error_msg)
            return {"error": True, "message": error_msg}