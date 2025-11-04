from typing import Union, Any, Dict
from pathlib import Path

import numpy as np
import torch
from datasets import tqdm
from diffusers import DDIMScheduler, UNet2DModel
from torch import Tensor

from model.src.model.base_model import IModel

class CatModel(IModel):
    def __init__(self, model_path: Union[str, Path], device: Union[str, Path, torch.device], timestamps: int = 25):
        super().__init__()
        self.timestamps = timestamps
        self.noise_scheduler = None
        self.model_path = Path(model_path)
        self.device = device
        self.model = torch.nn.Module()
        self.to_device(self.device)
        self.load_weights(self.model_path)

    @staticmethod
    def create_model(self, model_path: Union[str, Path], device: torch.device) -> IModel:
        md = CatModel(model_path, device)
        return md

    def load_weights(self, weights_path: Union[str, Path]) -> None:
        checkpoint = torch.load(weights_path, weights_only=False, map_location=self.device)
        self.model = UNet2DModel(**checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.load_schedular(checkpoint['scheduler_config'])
        self.model.to(self.device)

    def set_timestamps(self, timestamps: int) -> None:
        self.timestamps = timestamps

    def get_timestamps(self) -> int:
        return self.timestamps

    def load_schedular(self, config: Dict[str, Any]) -> None:
        self.noise_scheduler = DDIMScheduler.from_config(config)

    def preprocess(self, input_data: np.ndarray) -> np.ndarray:
        return input_data

    def postprocess(self, model_output: Tensor) -> np.ndarray:
        img = model_output.permute(0, 2, 3, 1).cpu().numpy()
        img = img * 0.5 + 0.5  # Денормализация
        img = np.clip(img, 0, 1)
        return img

    def get_torch_model(self):
        return self.model

    def get_device(self):
        return self.device

    def to_device(self, device: torch.device):
        self.device = device
        self.model.to(self.device)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_path': self.model_path,
            'device': self.device,
            'state_dict': self.model
        }

    def predict(self, input_data: np.ndarray, **kwargs) -> Any:
        with torch.no_grad():
            # Создаем случайный шум
            noise = torch.from_numpy(input_data).to(self.device)

            # Устанавливаем количество шагов генерации
            self.noise_scheduler.set_timesteps(self.timestamps)

            # Процесс денойзинга
            for t in tqdm(self.noise_scheduler.timesteps, desc="Final Generation"):
                residual = self.model(noise, t, return_dict=False)[0]
                noise = self.noise_scheduler.step(residual, t, noise).prev_sample

            generated_image = noise

        return generated_image
