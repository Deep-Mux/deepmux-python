import json

import numpy
import requests

from deepmux.config import BASE_URL
from deepmux.exceptions import ModelExceptionFactory
from deepmux.util import numpy_serialize_type


class APIInterface:
    def __init__(self, base_url: str = BASE_URL, timeout_sec: int = 60):
        self.base_url = base_url
        self.timeout_sec = timeout_sec

    def create(self, name: str, input_shape: list, output_shape: list, tensor_type: str, token: str):
        shapes_dict = {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'data_type': tensor_type,
        }
        resp = self._do_request(f'v1/model/{name}', method='PUT', json_dict=shapes_dict, token=token)

        return resp.json()

    def get(self, name: str, token: str = None):
        resp = self._do_request(f'v1/model/{name}', method='GET')

        return resp.json()

    def upload(self, name: str, model_file, token: str = None):
        if isinstance(model_file, str):
            # File path
            with open(model_file, 'rb') as file:
                data = file.read()
        else:
            # File-like object
            data = model_file.getvalue()
        resp = self._do_request(f'v1/model/{name}', method='POST', data=data, token=token)
        return resp.json()

    def run(self, model: str, tensor: numpy.ndarray, output_shape: list = None, token: str = None):
        if output_shape is None:
            output_shape = [1, -1]
        files = {
            'tensor': tensor.tobytes()
        }
        payload = {
            'shape': json.dumps(list(tensor.shape)),
            'data_type': numpy_serialize_type(tensor.dtype),
        }
        resp = self._do_request(f'v1/model/{model}/run', method='POST', data=payload, files=files, token=token)
        return numpy.frombuffer(resp.content, dtype=numpy.float32).reshape(output_shape)

    def _do_request(self, endpoint: str, method: str, data: dict = None, json_dict: dict = None, files: dict = None,
                    token: str = None):

        if data is None:
            data = dict()

        if files is None:
            files = dict()

        if json_dict is None:
            json_dict = dict()

        headers = {
            'X-Token': token
        }

        url = f"{self.base_url}/{endpoint}"

        resp = requests.request(url=url, method=method, data=data,json=json_dict, files=files,
                                timeout=self.timeout_sec, headers=headers)

        if resp.status_code != 200:
            response = resp.json()
            error_message = response.get('message')
            raise ModelExceptionFactory.get_exception_by_code(resp.status_code, error_message)

        return resp
