from enum import Enum

import numpy

from deepmux.exceptions import ModelStateError, ModelProcessingError
from deepmux.interface import APIInterface
from deepmux.config import BASE_URL


class ModelState(Enum):
    CREATED = 1
    PROCESSING = 2
    READY = 3
    ERROR = 4
    UNKNOWN = 5


class Model:

    def __init__(self, name: str, state: ModelState, input_shape: numpy.array, output_shape: numpy.array,
                 data_type: str, token: str, service_url: str = BASE_URL):
        self.name = name
        self.state = state.value
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_type = data_type
        self.interface = APIInterface(service_url)
        self.token = token

    def run(self, tensor: numpy.ndarray):
        """
        Run current model using tensor
        """
        self.state = getattr(ModelState, self.interface.get(self.name, token=self.token).get('state')).value
        if self.state == ModelState.PROCESSING.value:
            raise ModelProcessingError('Model is processing. Please try again later.')
        elif self.state != ModelState.READY.value:
            raise ModelStateError('You are not allowed to run non-ready model')
        return self.interface.run(model=self.name, tensor=tensor, output_shape=self.output_shape, token=self.token)

    def __repr__(self):
        return 'Model(name={}, state={}, input_shape={}, output_shape={}, data_type={})'.\
            format(self.name, self.state, self.input_shape, self.output_shape, self.data_type)
