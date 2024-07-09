from flask import Flask, request, jsonify
import chatbot  # Import the converted notebook script

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Food Delivery Platform"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Call your notebook functions here
    result = converted_notebook.some_function(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
