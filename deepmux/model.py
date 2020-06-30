from enum import Enum
from typing import Union, List
import numpy

from deepmux.exceptions import ModelStateError, ModelProcessingError
from deepmux.interface import APIInterface
from deepmux.config import BASE_URL
from deepmux.util import numpy_parse_type

class ModelState(Enum):
    CREATED = 1
    PROCESSING = 2
    READY = 3
    ERROR = 4
    UNKNOWN = 5


class Model:

    def __init__(self, name: str, state: ModelState, input_shape: numpy.array, output_shape: numpy.array,
            data_type: str, error: str, token: str, service_url: str = BASE_URL):
        self.name = name
        self.state = state.value
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data_type = numpy_parse_type(data_type)
        self.interface = APIInterface(service_url)
        self.token = token
        self.error = error

    def run(self, *tensors: numpy.ndarray) -> List[numpy.ndarray]:
        """
        Run current model using tensor
        Shape of tensor must be equal to model's input shape
        :param tensor: Input tensor, numpy array with shape that equals input_shape given while creating model e.g.
        [1, 3, 227, 227]. If model has multiple inputs, input tensors should be passed as positional arguments.
        Array’s dtype must match model’s dtype
        :return Model output. Shape of model output is equal to model's output shape
        """
        if self.state != ModelState.READY.value:
            model = self.interface.get(self.name, token=self.token)
            self.state = getattr(ModelState, model.get('state')).value
            self.error = model.get('error')

        if self.state == ModelState.PROCESSING.value:
            raise ModelProcessingError('Model is processing. Please try again later.')
        elif self.state == ModelState.ERROR.value:
            raise ModelStateError(f'Model validation finished with an error: {self.error}')
        elif self.state != ModelState.READY.value:
            raise ModelStateError('You can\'t run non-ready model')

        return self.interface.run(model=self.name, tensors=tensors, output_shape=self.output_shape,
                                  data_type=self.data_type, token=self.token)

    def __repr__(self):
        return 'Model(name={}, state={}, input_shape={}, output_shape={}, data_type={})'. \
            format(self.name, self.state, self.input_shape, self.output_shape, self.data_type)
