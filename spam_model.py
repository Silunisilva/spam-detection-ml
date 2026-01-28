import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert labels to numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Interactive testing
while True:
    text = input("\nEnter SMS (or type 'exit'): ")
    if text.lower() == "exit":
        break
    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)[0]
    print("Prediction:", "SPAM 🚫" if result == 1 else "HAM ✅")
