import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ----------------------------------
# 1. Load datasets
# ----------------------------------
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df[["text", "label"]]
df["text"] = df["text"].fillna("")

# ----------------------------------
# 2. Text cleaning
# ----------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

# ----------------------------------
# 3. Train-test split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ----------------------------------
# 4. Pipeline
# ----------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# ----------------------------------
# 5. Hyperparameter Grid
# ----------------------------------
param_grid = {
    "tfidf__max_df": [0.7],
    "tfidf__ngram_range": [(1, 1)],   # remove bigrams
    "tfidf__max_features": [5000],
    "clf__C": [0.1, 1]
}


# ----------------------------------
# 6. Grid Search
# ----------------------------------
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring="f1"
)

grid.fit(X_train, y_train)

# ----------------------------------
# 7. Evaluation with Metrics
# ----------------------------------
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("\nBest Parameters:")
for k, v in grid.best_params_.items():
    print(f"{k}: {v}")

print("\nEvaluation Metrics:\n")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------
# 8. Save model
# ----------------------------------
pickle.dump(best_model, open("model/fake_news_model.pkl", "wb"))

print("\nFine-tuned model saved successfully!")
