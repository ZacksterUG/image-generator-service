from pathlib import Path
from typing import Union, Dict, Optional, Type
from enum import Enum
from abc import ABC, abstractmethod

import torch

from model.src.model.base_model import IModel
from model.src.model.concrete_models.butterfly_model import ButterflyModel
from model.src.model.concrete_models.cat_model import CatModel


class ModelType(Enum):
    """Типы поддерживаемых моделей"""
    BUTTERFLY = "butterfly"
    CAT = "cat"
    #DOG = "dog"


class ModelFactory:
    """
    Фабричный класс для создания экземпляров моделей.
    """

    # Реестр доступных моделей
    _model_registry: Dict[ModelType, Type[IModel]] = {
        ModelType.BUTTERFLY: ButterflyModel,
        ModelType.CAT: CatModel,
        #ModelType.DOG: DogModel,
    }

    # Сопоставление строковых имен с типами моделей
    _name_to_type: Dict[str, ModelType] = {
        "butterfly": ModelType.BUTTERFLY,
        "cat": ModelType.CAT,
        #"dog": ModelType.DOG,
        "butterfly_model": ModelType.BUTTERFLY,
        "cat_model": ModelType.CAT,
        #"dog_model": ModelType.DOG,
    }

    @classmethod
    def register_model(cls, model_type: ModelType, model_class: Type[IModel]) -> None:
        """
        Регистрирует новую модель в фабрике.

        Args:
            model_type: Тип модели
            model_class: Класс модели, реализующий IModel
        """
        if not issubclass(model_class, IModel):
            raise ValueError(f"Класс {model_class} должен реализовывать интерфейс IModel")

        cls._model_registry[model_type] = model_class
        print(f"Модель {model_type.value} зарегистрирована")

    @classmethod
    def unregister_model(cls, model_type: ModelType) -> None:
        """
        Удаляет модель из реестра фабрики.

        Args:
            model_type: Тип модели для удаления
        """
        if model_type in cls._model_registry:
            del cls._model_registry[model_type]
            print(f"Модель {model_type.value} удалена из реестра")

    @classmethod
    def get_available_models(cls) -> list:
        """
        Возвращает список доступных типов моделей.

        Returns:
            Список доступных ModelType
        """
        return list(cls._model_registry.keys())

    @classmethod
    def create_model(cls,
                     model_type: Union[ModelType, str],
                     model_path: Union[str, Path],
                     device: Optional[Union[torch.device, str]] = None) -> IModel:
        """
        Создает экземпляр модели указанного типа.

        Args:
            model_type: Тип модели (ModelType или строковое имя)
            model_path: Путь к файлу с весами модели
            device: Устройство для загрузки модели (по умолчанию CPU)

        Returns:
            Экземпляр модели, реализующий IModel

        Raises:
            ValueError: Если тип модели не поддерживается
            FileNotFoundError: Если файл модели не найден
        """
        # Нормализация типа модели
        if isinstance(model_type, str):
            model_type = model_type.lower()
            if model_type not in cls._name_to_type:
                raise ValueError(f"Неизвестный тип модели: {model_type}. "
                                 f"Доступные: {list(cls._name_to_type.keys())}")
            model_type = cls._name_to_type[model_type]

        # Проверка существования файла модели
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        # Устройство по умолчанию
        if device is None:
            device = torch.device('cpu')

        # Создание модели
        if model_type not in cls._model_registry:
            raise ValueError(f"Модель типа {model_type} не зарегистрирована в фабрике")

        model_class = cls._model_registry[model_type]
        return model_class.create_model(None, model_path, device)

    @classmethod
    def create_model_from_config(cls, config: Dict) -> IModel:
        """
        Создает модель на основе конфигурационного словаря.

        Args:
            config: Словарь с конфигурацией {
                'type': 'butterfly' | 'cat' | 'dog',
                'model_path': '/path/to/model.pth',
                'device': 'cuda' | 'cpu' (опционально)
            }

        Returns:
            Экземпляр модели
        """
        model_type = config['type']
        model_path = config['model_path']
        device = torch.device(config.get('device', 'cpu'))

        return cls.create_model(model_type, model_path, device)
