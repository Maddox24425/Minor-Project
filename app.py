import joblib

# 1. Load the entire trained pipeline from the file
# This only needs to be done once when your application starts.
print("Loading sentiment analysis pipeline...")
model_pipeline = joblib.load('lr_sentiment_pipeline.joblib')
print("✅ Pipeline loaded successfully.")

# 2. Define the text you want to analyze
# This would be the input you get from your user in a real application.
new_sentence_positive = "I had a wonderful time at the event, it was absolutely fantastic!"
new_sentence_negative = "The movie was a complete disaster and a total waste of my time."

# 3. Use the pipeline's .predict() method directly on the raw text
# The pipeline automatically handles the vectorization internally.
# The input must be a list or an iterable.
prediction_positive = model_pipeline.predict([new_sentence_positive])
prediction_negative = model_pipeline.predict([new_sentence_negative])

# The output is a NumPy array, where 0 is negative and 1 is positive.
print(f"\nText: '{new_sentence_positive}'")
print(f"Predicted Sentiment (0=Negative, 1=Positive): {prediction_positive[0]}")

print(f"\nText: '{new_sentence_negative}'")
print(f"Predicted Sentiment (0=Negative, 1=Positive): {prediction_negative[0]}")

# 4. Creating a user-friendly function
def predict_sentiment(text):
    """
    Takes a string of text and returns a user-friendly sentiment prediction.
    """
    # Make sure the input is in a list for the pipeline
    prediction = model_pipeline.predict([text])
    
    # Interpret the numerical output
    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"

# Example using the function
print("\n--- Using the prediction function ---")
sentiment = predict_sentiment("The customer service was incredibly helpful and friendly.")
print(f"The predicted sentiment is: {sentiment}")