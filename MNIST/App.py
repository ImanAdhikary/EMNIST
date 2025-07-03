from flask import Flask, request, jsonify, render_template
from model_def import NeuralNetwork
import torch
import numpy as np
import os   # ✅ Import os here

# Set up model architecture
model = NeuralNetwork(num_features=784)

# ✅ Safely construct absolute path to model file relative to App.py
model_path = os.path.join(os.path.dirname(__file__), 'fashion_mnist_model.pth')

# Load model weights
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluation mode

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = request.form['pixel_values']
        pixel_values = [float(x) for x in input_values.split(',')]
        input_tensor = torch.tensor([pixel_values], dtype=torch.float32)
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        return render_template('index.html', prediction_text=f'Predicted Class: {predicted_class}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
