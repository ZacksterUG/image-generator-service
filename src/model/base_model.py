from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from pathlib import Path



class IModel(ABC):
    """
    Базовый интерфейс для моделей нейронных сетей.
    Определяет контракт для загрузки весов, предсказания и доступа к низкоуровневым операциям.
    """

    @abstractmethod
    def load_weights(self, weights_path: Union[str, Path]) -> None:
        """
        Загружает веса модели из файла.

        Args:
            weights_path: Путь к файлу с весами модели

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат файла не поддерживается
        """
        pass

    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Any:
        """
        Выполняет предсказание на входных данных.

        Args:
            input_data: Входные данные для модели
            **kwargs: Дополнительные параметры для предсказания

        Returns:
            Результат предсказания модели
        """
        pass

    @abstractmethod
    def get_torch_model(self) -> nn.Module:
        """
        Возвращает экземпляр PyTorch модели для низкоуровневых операций.

        Returns:
            Экземпляр torch.nn.Module
        """
        pass

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """
        Предобработка входных данных перед подачей в модель.

        Args:
            input_data: Сырые входные данные

        Returns:
            Предобработанные данные готовые для модели
        """
        pass

    @abstractmethod
    def postprocess(self, model_output: Any) -> Any:
        """
        Постобработка выхода модели в финальный результат.

        Args:
            model_output: Сырой выход модели

        Returns:
            Обработанный результат предсказания
        """
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        """
        Возвращает устройство, на котором находится модель.

        Returns:
            torch.device: Устройство модели (CPU/GPU)
        """
        pass

    @abstractmethod
    def to_device(self, device: Union[str, torch.device]) -> None:
        """
        Перемещает модель на указанное устройство.

        Args:
            device: Устройство для перемещения ('cpu', 'cuda', torch.device)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о модели.

        Returns:
            Словарь с информацией о модели (архитектура, версия, параметры и т.д.)
        """
        pass
    @staticmethod
    @abstractmethod
    def create_model(self, model_path: Union[str, Path], device: torch.device) -> 'IModel':
        pass