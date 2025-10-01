import joblib

print("Loading sentiment analysis pipeline...")
model_pipeline = joblib.load(r'D:\Data Science\Minor-Project\lr_sentiment_pipeline.joblib')
print("Pipeline loaded successfully.")

def predict_sentiment(text):
    prediction = model_pipeline.predict([text])
    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"

text=input("Enter Text Here")
sentiment = predict_sentiment(text)
print(f"The predicted sentiment is: {sentiment}")