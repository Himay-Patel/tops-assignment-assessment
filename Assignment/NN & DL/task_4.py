import pandas as pd

df = pd.read_csv("combined_emotion.csv").rename(columns={"sentence":"text","emotion":"label"})

# sample smaller for demonstration
df_small = df.sample(20000, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    df_small["text"], df_small["label"], test_size=0.2, random_state=42, stratify=df_small["label"]
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
    ("clf", SGDClassifier(loss="log_loss"))
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)

acc
