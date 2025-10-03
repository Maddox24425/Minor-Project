from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

# 1. Initialize the Flask application
app = Flask(__name__)

# Enable CORS to allow your frontend to make requests to this API
CORS(app)

# 2. Load your trained model pipeline
# This path should be the absolute path to your model file.
MODEL_PATH = r'D:\Data Science\Minor-Project\lr_sentiment_pipeline.joblib'
model_pipeline = None  # Initialize as None

try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# 3. Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """Receives text input via a POST request and returns a sentiment prediction."""

    # Simple check to make sure the model loaded correctly
    if model_pipeline is None:
        return jsonify({'error': 'Model is not loaded, check the server logs.'}), 500

    try:
        # Get the data from the POST request's body
        data = request.get_json()
        text_input = data['text']

        # The model expects a list, so we put the input text in a list
        prediction = model_pipeline.predict([text_input])

        # Convert the numerical prediction (0 or 1) to a readable string
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

        # Return the result as a JSON response
        return jsonify({'sentiment': sentiment})

    except Exception as e:
        # A simple catch-all for any errors that might occur
        return jsonify({'error': str(e)}), 400

# 4. Run the Flask application
if __name__ == '__main__':
    # This starts the server on http://127.0.0.1:5000
    app.run(port=5000, debug=True)

