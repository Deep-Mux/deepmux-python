# ResNet classifier for ImageNet example

This example shows the complete proccess of creating a model and uploading it to deepmux servers for inference.

There are two key files in this example:
* `create_model.ipynb` – Notebook that sets up the model and uploads it to deepmux;
* `serve.py` - Simple Flask server that serves your model

Walkthrough for this example is given below

All dependencies for this example are stored in `requirements.txt`. If you miss anything, simply run 
```shell script
pip install -r requirements.txt
```

### Step one: create model

Open `create_model.ipynb` notebook and follow instructions provided in the notebook.

You can open the notebook by typing 
```shell script
jupyter notebook create_model.ipynb
```

Once you see the message `Model uploaded, ready to serve` appear, you are ready to proceed to the next step.

### Step two: run server

Open `serve.py` in your favorite editor and replace `<YOUR TOKEN HERE>` your token.

Once you are done, run the server by typing 
```shell script
python serve.py
```

Now you can query your server and get predictions. We will use `curl` command line utility to send requests.
Open a second terminal and type
```shell script
 curl http://localhost:8000/classify -X POST -F "image=@images/balloon.jpg"
```
You should see the response appear:
```json
{"class":"balloon"}
```

Success! You can also try other images in `images` directory 
