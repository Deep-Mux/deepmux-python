from io import BytesIO

import numpy

from deepmux.interface import APIInterface
from deepmux.model import Model, ModelState
from deepmux.util import torch_serialize_type


def create_model(
        pytorch_model,
        model_name: str,
        input_shape: list,
        output_shape: list,
        token: str
) -> Model:
    """
    Creates model from pytorch model
    :param pytorch_model: torch.nn.Module object
    :param model_name: name of model
    :param input_shape: shape of input data
    :param output_shape: shape of output data
    :param token: Your token
    :return: Model class object
    """
    try:
        import torch
    except ImportError:
        raise ImportError('You need pytorch module for creating models')
    client = APIInterface()
    # Exporting model to ONNX format
    model_file = BytesIO()
    torch.onnx.export(pytorch_model,
                      torch.zeros(input_shape),
                      model_file,
                      input_names=['in'],
                      output_names=['out'])
    # Creating model on server
    tensor_type = torch_serialize_type(next(pytorch_model.parameters()).dtype)
    client.create(model_name, input_shape, output_shape, tensor_type, token=token)
    result = client.upload(model_name, model_file, token=token)
    return Model(name=result.get('name'),
                 state=getattr(ModelState, result.get('state')),
                 input_shape=numpy.array(result.get('input_shape')),
                 output_shape=numpy.array(result.get('output_shape')),
                 data_type=result.get('data_type'),
                 token=token)


def get_model(model_name: str, token: str) -> Model:
    """
    Fetch model by name
    :param model_name: name of Model
    :param token: Your token
    :return: Model class object
    """
    client = APIInterface()
    result = client.get(model_name, token=token)
    return Model(name=result.get('name'),
                 state=getattr(ModelState, result.get('state')),
                 input_shape=numpy.array(result.get('input_shape')),
                 output_shape=numpy.array(result.get('output_shape')),
                 data_type=result.get('data_type'),
                 token=token)
