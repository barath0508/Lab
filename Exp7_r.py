import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Read dataset
data = pd.read_csv("emails_exp7.csv")

# Clean text
stop = set(stopwords.words("english"))

def clean(text):
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(w for w in text.split() if w.lower() not in stop)

data["clean_text"] = data["text"].apply(clean)

# Convert text into numbers
cv = CountVectorizer()
X = cv.fit_transform(data["clean_text"])
y = data["spam"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
