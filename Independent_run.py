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
    
while True:
    
    text=input("Enter Text Here: ")
    sentiment = predict_sentiment(text)
    print(f"The predicted sentiment is: {sentiment}")
    inputt=int(input('Do you want to continue?\nPress 1 to continue\nPress 2 to Exit\n'))
    if inputt == 1:
        continue
    elif inputt==2:
        break
    else:
        print('Invalid Input')
        break

print("Thanks For using")
    
