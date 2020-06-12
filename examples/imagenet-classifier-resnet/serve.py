import requests
from flask import Flask, jsonify, request
from deepmux import get_model
import numpy as np
from torchvision import transforms
from PIL import Image

# Paste your token here
token = '<YOUR TOKEN HERE>'

# Get model reference
model = get_model(model_name='imagenet-classifier-resnet', token=token)

# Load imagenet class labels
classes = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json').json()
classes = dict((int(k), v[1]) for k, v in classes.items())


# Setup Flask app
app = Flask('imagenet-classifier-resnet')


# Init preprocessor
preprocessor = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess(image_file):
    input_image = Image.open(image_file)
    input_tensor = preprocessor(input_image)
    return input_tensor.unsqueeze(0).numpy()


@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            raise ValueError('No image parameter in request')
        image = request.files['image']

        tensor = preprocess(image)

        result = model.run(tensor)

        return jsonify({
            'class': classes[np.argmax(result[0])]
        })
    except Exception as e:
        return jsonify({
            'error': repr(e)
        }), 400


if __name__ == '__main__':
    app.run('0.0.0.0', 8000)
