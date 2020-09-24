import torch
from classifier_api.Model import IrisModel
from flask import Flask, request
from flasgger  import Swagger
import os

os.environ["FLASK_ENV"] = "development"
app = Flask(__name__)
Swagger(app)

new_model = IrisModel(4, 3, 8, 9)
new_model.load_state_dict(torch.load('./IrisDatasetModel.pt'))
new_model.eval()


@app.route('/')
def greet():
    return "Welcome to the Iris Classifier"


@app.route('/predict', methods=["Get"])
def predict_class():
    """Lets predict the Iris Class
    This is using docstrings for specifications.
    ---
    parameters:
      - name: sepal_length
        in: query
        type: number
        required: true

      - name: sepal_width
        in: query
        type: number
        required: true

      - name: petal_length
        in: query
        type: number
        required: true

      - name: petal_width
        in: query
        type: number
        required: true

    responses:
      200:
        description: The output values
    """
    sepal_length = float(request.args.get('sepal_length'))
    sepal_width = float(request.args.get('sepal_width'))
    petal_length = float(request.args.get('petal_length'))
    petal_width = float(request.args.get('petal_width'))
    input_tensor = torch.tensor([sepal_length, sepal_width, petal_length, petal_width])
    target_pred = torch.argmax(new_model(input_tensor))
    labels = ['Iris setosa', 'Iris virginica', 'Iris versicolor']
    prediction = labels[target_pred]
    return "The predicted class is " + str(prediction)


if __name__ == "__main__":
    print("name: ", __name__)
    app.run(host='0.0.0.0', port=8000)
