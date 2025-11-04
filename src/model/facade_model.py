from typing import Union, Any

import torch

from .base_model import IModel

class FacadeModel:
    def __init__(self, cat_model: IModel, butterfly_model: IModel) -> None:
        self.cat_model = cat_model
        self.butterfly_model = butterfly_model

    def to_device(self, device: Union[str, torch.device]):
        self.cat_model.to_device(device)
        self.butterfly_model.to_device(device)

    def generate_by_class(self, cl: str, input_data: Any):
        if cl == 'cat':
            return self.generate_cat(input_data)
        elif cl == 'butterfly':
            return self.generate_butterfly(input_data)

        raise Exception('Unknown class')

    def generate_cat(self, input_data: Any):
        preprocess = self.cat_model.preprocess(input_data)
        prediction = self.cat_model.predict(preprocess)
        postprocess = self.cat_model.postprocess(prediction)
        return postprocess

    # def generate_dog(self, input_data: Any):
    #     preprocess = self.dog_model.preprocess(input_data)
    #     prediction = self.dog_model.predict(preprocess)
    #     postprocess = self.dog_model.postprocess(prediction)
    #     return postprocess

    def generate_butterfly(self, input_data: Any):
        preprocess = self.butterfly_model.preprocess(input_data)
        prediction = self.butterfly_model.predict(preprocess)
        postprocess = self.butterfly_model.postprocess(prediction)
        return postprocess

    def get_cat_model(self):
        return self.cat_model

    # def get_dog_model(self):
    #     return self.dog_model

    def get_butterfly(self):
        return self.butterfly_model


