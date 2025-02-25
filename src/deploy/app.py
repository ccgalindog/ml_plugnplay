import pandas as pd
from flask import Flask, request, jsonify
from src.pipelines.inference import InferencePipeline


app = Flask(__name__)

inference_pipeline = InferencePipeline(model_path='model',
                                       artifacts_path='artifacts')

# Start an app and define an endpoint in the /predict method
# to perform inferences with the model over the received data request
@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    df_prod = pd.json_normalize(request.json)
    predictions = inference_pipeline.get_inference(df_prod)
    
    return jsonify({'predicted_value' : list(predictions)})
