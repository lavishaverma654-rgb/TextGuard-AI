import argparse
from pathlib import Path

import joblib


def main():
    parser = argparse.ArgumentParser(description="Classify one SMS message with TextGuard.")
    parser.add_argument("message", help="SMS text to classify.")
    parser.add_argument("--model", default="textguard_model.joblib", help="Saved model path.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model not found at {model_path}. Train it first using train.py."
        )

    model = joblib.load(model_path)
    prediction = model.predict([args.message])[0]
    probabilities = model.predict_proba([args.message])[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(prediction)])

    print(f"Prediction: {prediction.upper()}")
    print(f"Model confidence: {confidence:.1%}")
    print("Note: This is an educational prediction, not a guarantee.")


if __name__ == "__main__":
    main()
