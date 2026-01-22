import pickle

# Load full pipeline model (TF-IDF + Classifier)
model = pickle.load(open("model/fake_news_model.pkl", "rb"))

# Sample news text
news_text = """
Breaking news! Scientists confirm water cures all diseases instantly.
"""

# Safety check
if not news_text.strip():
    print("Error: News text is empty.")
    exit()

# Prediction (NO manual vectorization needed)
prediction = model.predict([news_text])[0]

# Confidence score
confidence = model.predict_proba([news_text])[0]
confidence_score = confidence[prediction] * 100

# Output
label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

print("Prediction :", label)
print(f"Confidence : {confidence_score:.2f}%")
