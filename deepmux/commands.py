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
        token: str,
) -> Model:
    """
    Creates model from pytorch model
    :param pytorch_model: torch.nn.Module object
    :param model_name: name of model
    How to define dynamic axes?
    You should put 0 instead of shape value.
    Example: [0, 1, 64] - first axe will be dynamic
    :param input_shape: shape of input data
    :param output_shape: shape of output data
    :param token: Your token
    :return: Model class object
    """
    if not len(input_shape) or not len(output_shape):
        raise ValueError('Input and output shapes can\'t be empty')

    if not isinstance(input_shape[0], (tuple, list, numpy.ndarray)):
        input_shape = [input_shape]

    if not isinstance(output_shape[0], (tuple, list, numpy.ndarray)):
        output_shape = [output_shape]

    try:
        import torch
    except ImportError:
        raise ImportError('You need pytorch module for creating models')
    client = APIInterface()
    # Exporting model to ONNX format
    model_file = BytesIO()
    # Dynamic axes
    input_dynamic_axes = [[] for i in range(len(input_shape))]
    output_dynamic_axes = [[] for i in range(len(output_shape))]

    dynamic_axes_dict = {}

    def fill_dynamic_axes(dynamic_axes, io_shape, io_prefix, result_dict):
        for idx, shape in enumerate(io_shape):
            for axe_idx, axe_value in enumerate(shape):
                if axe_value == 0:
                    dynamic_axes[idx].append(axe_idx)
            if dynamic_axes[idx]:
                dynamic_axes_dict[f'{io_prefix}_{idx}'] = dynamic_axes[idx]

    fill_dynamic_axes(input_dynamic_axes, input_shape, 'in', dynamic_axes_dict)
    fill_dynamic_axes(output_dynamic_axes, output_shape, 'out', dynamic_axes_dict)

    sample_input_shape = [[1 if x == 0 else x for x in shape] for shape in input_shape]
    torch.onnx.export(pytorch_model,
                      tuple(torch.zeros(x) for x in sample_input_shape),
                      model_file,
                      input_names=[f'in_{x}' for x in range(len(input_shape))],
                      output_names=[f'out_{x}' for x in range(len(output_shape))],
                      dynamic_axes=dynamic_axes_dict,
                      opset_version=11)
    # Creating model on server
    tensor_type = torch_serialize_type(next(pytorch_model.parameters()).dtype)
    client.create(model_name, input_shape, output_shape, tensor_type, token=token)
    result = client.upload(model_name, model_file, token=token)
    return Model(name=result.get('name'),
                 state=getattr(ModelState, result.get('state')),
                 input_shape=numpy.array(result.get('input')),
                 output_shape=numpy.array(result.get('output')),
                 data_type=result.get('data_type'),
                 error=result.get('error'),
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
                 input_shape=numpy.array(result.get('input')),
                 output_shape=numpy.array(result.get('output')),
                 data_type=result.get('data_type'),
                 error=result.get('error'),
                 token=token)
