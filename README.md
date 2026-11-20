# TextGuard – SMS Spam Detection Using Machine Learning

**Building AI course project**

## Summary
TextGuard is a beginner-friendly machine-learning project that classifies an SMS message as **spam** or **ham (not spam)**. It uses text preprocessing and a Multinomial Naive Bayes classifier to learn patterns from labelled messages and predict the class of a new message.

## Background
Unwanted messages can contain scams, misleading offers, and suspicious links. Manually checking every message takes time, and people may overlook warning signs. TextGuard explores how a simple machine-learning model can help flag likely spam messages.

The project is an educational prototype. It does not guarantee that every message will be classified correctly and should not be used as the only protection against scams.

## How is it used?
1. Collect SMS messages labelled as `spam` or `ham`.
2. Clean and prepare the message text.
3. Split the labelled data into training and test sets.
4. Train a text-classification pipeline using TF-IDF features and Multinomial Naive Bayes.
5. Evaluate the model on held-out test messages.
6. Enter a new message and receive a predicted label.

## How to run
### 1. Install Python
Use Python 3.9 or newer.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add a dataset
The included `sample_data.csv` is a tiny demonstration dataset, not a reliable evaluation dataset. For meaningful results, download the **SMS Spam Collection** dataset from the UCI Machine Learning Repository and save it as `data/spam.csv`.

The script accepts either:
- a CSV with columns named `label` and `message`, or
- the common UCI SMS format with the first two columns containing the label and message.

### 4. Train and evaluate
```bash
python train.py --data data/spam.csv
```

### 5. Classify a message
```bash
python predict.py "Congratulations! You have won a free prize. Click now."
```

If you want to try the included small demo data:
```bash
python train.py --data sample_data.csv
```
Results from the tiny demo dataset are only for checking that the code runs.

## Project structure
```text
TextGuard/
├── README.md
├── requirements.txt
├── sample_data.csv
├── train.py
└── predict.py
```

## AI methods
- **TF-IDF:** converts message text into numerical features based on word importance.
- **Multinomial Naive Bayes:** learns word patterns associated with spam and ham.
- **Train/test split:** checks performance on messages not used for training.

## Challenges
- Spam messages change over time, so a model trained on old messages may become less accurate.
- Short messages can be ambiguous.
- A small or imbalanced dataset can produce misleading accuracy.
- False positives matter: a genuine message could be marked as spam.

## What next?
- Train and test on a larger, more diverse dataset.
- Compare Naive Bayes with Logistic Regression and other classifiers.
- Report precision, recall, F1-score, and a confusion matrix.
- Add a simple web interface.
- Explore multilingual messages and methods to reduce false positives.

## Acknowledgements
- UCI Machine Learning Repository: SMS Spam Collection dataset (use the original dataset terms and cite the dataset when using it).
- Scikit-learn documentation for text feature extraction and machine-learning models.

## Responsible use
TextGuard is a learning project, not a security product. Do not upload private or sensitive messages to a public demo. Always verify suspicious messages independently and never click unknown links or share passwords, OTPs, or banking details.
