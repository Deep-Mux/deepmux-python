import io
import json
from typing import Union, List, Optional, Tuple

import numpy
import requests

from deepmux.config import BASE_URL
from deepmux.exceptions import ModelExceptionFactory
from deepmux.util import numpy_serialize_type


class APIInterface:
    def __init__(self, base_url: str = BASE_URL, timeout_sec: int = 600):
        self.base_url = base_url
        self.timeout_sec = timeout_sec

    def create(self,
               name: str,
               input_shape: List[Union[List[int], Tuple[int]]],
               output_shape: List[Union[List[int], Tuple[int]]],
               tensor_type: str,
               token: str):
        shapes_dict = {
            'input': input_shape,
            'output': output_shape,
            'data_type': tensor_type,
        }

        resp = self._do_request(f'v1/model/{name}', method='PUT', json_dict=shapes_dict, token=token)

        return resp.json()

    def get(self, name: str, token: str = None):
        resp = self._do_request(f'v1/model/{name}', method='GET', token=token)

        return resp.json()

    def upload(self,
               name: str, model_file: io.BytesIO, token: str):
        data = model_file.getvalue()
        resp = self._do_request(f'v1/model/{name}', method='POST', data=data, token=token)
        return resp.json()

    def run(self, model: str,
            tensors: Union[numpy.ndarray, List[numpy.ndarray], Tuple[numpy.ndarray]],
            output_shape: List[Union[List[int], Tuple[int]]],
            data_type: type,
            token: str = None):

        files = {
                'tensor': b''.join(map(lambda x: x.tobytes(), tensors))
        }
        payload = {
            'shape': json.dumps(list(map(lambda x: x.shape, tensors))),
            'data_type': numpy_serialize_type(tensors[0].dtype),
        }
        resp = self._do_request(f'v1/model/{model}/run', method='POST', data=payload, files=files, token=token)
        resp_bytes = resp.content
        if len(output_shape) == 1:
            return numpy.frombuffer(resp_bytes, dtype=data_type).reshape(output_shape[0])
        else:
            result = []
            total_size = 0
            for shape in output_shape:
                tensor_size = numpy.prod(shape) * data_type().itemsize
                cur_buffer = resp_bytes[total_size: total_size + tensor_size]
                result.append(numpy.frombuffer(cur_buffer, dtype=data_type).reshape(shape))
                total_size += tensor_size
            return result

    def _do_request(self, endpoint: str,
                    method: str,
                    token: str,
                    data: Optional[Union[dict, bytes]] = None,
                    json_dict: Optional[dict] = None,
                    files: Optional[dict] = None):

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

        resp = requests.request(url=url, method=method, data=data, json=json_dict, files=files,
                                timeout=self.timeout_sec, headers=headers)

        if resp.status_code != 200:
            response = resp.json()
            error_message = response.get('message')
            raise ModelExceptionFactory.get_exception_by_code(resp.status_code, error_message)

        return resp