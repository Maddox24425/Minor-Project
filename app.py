from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1. Initialize the Flask application
app = Flask(__name__)

# Configure CORS more securely (adjust origins as needed)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])  # Add your Node.js server URL

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', r'D:\Data Science\Minor-Project\lr_sentiment_pipeline.joblib')
MAX_TEXT_LENGTH = 10000  # Maximum allowed text length

# 2. Load your trained machine learning pipeline
def load_model():
    """Load the sentiment analysis model pipeline."""
    try:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            return None
            
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Load model at startup
model_pipeline = load_model()

# 3. Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running."""
    status = 'healthy' if model_pipeline is not None else 'unhealthy'
    return jsonify({
        'status': status,
        'model_loaded': model_pipeline is not None
    }), 200 if status == 'healthy' else 503

# 4. Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a POST request with JSON data containing 'text' 
    and returns a sentiment prediction.
    
    Expected JSON format:
    {
        "text": "Your text to analyze"
    }
    
    Response format:
    {
        "sentiment": "Positive" or "Negative",
        "confidence": float (optional, if available from model)
    }
    """
    # Check if model is loaded
    if model_pipeline is None:
        logger.error("Model is not loaded")
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 503

    try:
        # Validate request has JSON content
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        # Get the JSON data from the request body
        json_data = request.get_json()
        
        # Validate required fields
        if not json_data:
            return jsonify({'error': 'Request body cannot be empty'}), 400
            
        if 'text' not in json_data:
            return jsonify({'error': "Missing 'text' field in request body"}), 400
            
        text = json_data.get('text', '')
        
        # Validate text input
        if not isinstance(text, str):
            return jsonify({'error': "'text' must be a string"}), 400
            
        if not text.strip():
            return jsonify({'error': "'text' cannot be empty"}), 400
            
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({'error': f"'text' exceeds maximum length of {MAX_TEXT_LENGTH} characters"}), 400

        # The pipeline expects a list of texts
        text_to_predict = [text]

        # Make a prediction using the loaded pipeline
        prediction = model_pipeline.predict(text_to_predict)
        
        # Convert prediction to readable format
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        # Try to get prediction probabilities if available
        response = {'sentiment': sentiment}
        
        try:
            # If the pipeline supports predict_proba, include confidence score
            probabilities = model_pipeline.predict_proba(text_to_predict)
            confidence = float(max(probabilities[0]))
            response['confidence'] = round(confidence, 4)
        except (AttributeError, Exception):
            # Model doesn't support probability predictions or error occurred
            pass
        
        logger.info(f"Prediction made: {sentiment} for text of length {len(text)}")
        return jsonify(response), 200

    except Exception as e:
        # Handle any unexpected errors during prediction
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred during prediction. Please try again.'}), 500

# 5. Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# 6. Run the Flask application
if __name__ == '__main__':
    # Configuration for production vs development
    is_production = os.environ.get('FLASK_ENV') == 'production'
    
    if is_production:
        # In production, use a production WSGI server like Gunicorn
        logger.info("Running in production mode")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        # Development mode
        logger.info("Running in development mode")
        app.run(host='127.0.0.1', port=5000, debug=True)

