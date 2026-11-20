import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin-1")

    # Accept either descriptive column names or the common UCI SMS CSV layout.
    if {"label", "message"}.issubset(df.columns):
        data = df[["label", "message"]].copy()
    elif len(df.columns) >= 2:
        data = df.iloc[:, :2].copy()
        data.columns = ["label", "message"]
    else:
        raise ValueError("CSV must contain label and message columns.")

    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["spam", "ham"])]
    data = data.dropna().drop_duplicates()
    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both 'spam' and 'ham' labels.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Train the TextGuard SMS classifier.")
    parser.add_argument("--data", default="data/spam.csv", help="Path to labelled CSV data.")
    parser.add_argument("--model", default="textguard_model.joblib", help="Output model path.")
    args = parser.parse_args()

    data = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("classifier", MultinomialNB()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    output = Path(args.model)
    joblib.dump(model, output)
    print(f"\nSaved trained model to: {output.resolve()}")


if __name__ == "__main__":
    main()
