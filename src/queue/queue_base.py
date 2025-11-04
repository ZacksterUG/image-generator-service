from abc import ABC, abstractmethod
from typing import Any

class Message:
    """Контейнер для сообщения и его метаданных."""
    def __init__(self, body: Any, delivery_tag: Any = None):
        self.body = body
        self.delivery_tag = delivery_tag

class QueueBase(ABC):
    """Базовый абстрактный класс для работы с очередями сообщений.

    Определяет интерфейс для взаимодействия с различными реализациями очередей
    (например, RabbitMQ, Redis, in-memory и т.д.). Все методы должны быть
    реализованы в дочерних классах.

    Примечание:
        Конкретная реализация может использовать внутреннее состояние
        (например, текущую очередь для метода `pop`), либо принимать имя очереди
        как параметр — это зависит от архитектуры подкласса.
    """
    @abstractmethod
    def declare_queue(self, queue: str) -> bool:
        """Объявляет (создаёт, если не существует) очередь с заданным именем.

        Args:
            queue (str): Имя очереди, которую необходимо объявить.

        Returns:
            bool: True, если очередь успешно объявлена или уже существует;
                  False в случае ошибки.
        """
        pass

    @abstractmethod
    def push(self, queue: str, data: Any) -> bool:
        """Помещает сообщение в указанную очередь.

        Args:
            queue (str): Имя очереди, в которую нужно отправить сообщение.
            data (Any): Данные для отправки. Должны быть сериализуемы
                        в соответствии с требованиями конкретной реализации.

        Returns:
            bool: True, если сообщение успешно отправлено; False в случае ошибки.
        """
        pass

    @abstractmethod
    def pop(self) -> Any:
        """Извлекает и удаляет одно сообщение из очереди.

        Примечание:
            Метод предполагает, что очередь для извлечения сообщения
            уже определена (например, через конструктор или последний вызов `declare_queue`).
            Это может быть уточнено в реализации.

        Returns:
            Any: Извлечённые данные. Если очередь пуста, может вернуть None
                 или вызвать исключение — зависит от реализации.
        """
        pass

    @abstractmethod
    def ack(self, delivery_tag: Any) -> bool:
        """Подтверждает успешную обработку сообщения.

        После вызова сообщение удаляется из очереди.
        """
        pass

    @abstractmethod
    def nack(self, delivery_tag: Any, requeue: bool = True) -> bool:
        """Отклоняет сообщение.

        Если requeue=True — сообщение возвращается в очередь.
        """
        pass

    @abstractmethod
    def empty(self) -> bool:
        """Проверяет, пуста ли очередь (или очереди, с которыми работает экземпляр).

        Returns:
            bool: True, если очередь пуста; False — если содержит хотя бы одно сообщение.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Закрывает соединение с брокером."""
        pass

    @abstractmethod
    def ping(self) -> dict:
        """Проверяет работоспособность соединения с RabbitMQ.

        Returns:
            dict: {"error": bool, "message": str}
        """
        pass
