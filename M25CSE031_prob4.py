"""
SPORT vs POLITICS Text Classification
Roll Number: M25CSE031

This program builds a text classification system that
distinguishes between SPORT and POLITICS news articles.

The system:
1. Loads a large real-world dataset (20 Newsgroups)
2. Applies multiple feature extraction techniques
3. Trains and evaluates three ML algorithms
4. Compares their performance
5. Allows interactive prediction on new text
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# STEP 1: Load and Prepare Dataset
# ==========================================================

def load_sport_politics_dataset():
    """
    Loads selected categories from the 20 Newsgroups dataset.

    SPORT categories:
        - rec.sport.baseball
        - rec.sport.hockey

    POLITICS categories:
        - talk.politics.misc
        - talk.politics.guns
        - talk.politics.mideast

    Returns:
        texts  -> list of documents
        labels -> SPORT or POLITICS
    """

    selected_categories = [
        "rec.sport.baseball",
        "rec.sport.hockey",
        "talk.politics.misc",
        "talk.politics.guns",
        "talk.politics.mideast"
    ]

    dataset = fetch_20newsgroups(
        subset="all",
        categories=selected_categories,
        remove=("headers", "footers", "quotes")
    )

    document_texts = dataset.data
    document_labels = []

    for target_index in dataset.target:
        category_name = dataset.target_names[target_index]

        if "sport" in category_name:
            document_labels.append("SPORT")
        else:
            document_labels.append("POLITICS")

    return document_texts, document_labels


# ==========================================================
# STEP 2: Model Evaluation Utility
# ==========================================================

def train_and_evaluate_model(vectorizer, classifier,
                             X_train, X_test, y_train, y_test):
    """
    Trains a model using the given vectorizer and classifier.
    Then prints classification report and confusion matrix.
    """

    # Convert raw text into numeric vectors
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # Train the classifier
    classifier.fit(X_train_vectors, y_train)

    # Predict on test set
    predictions = classifier.predict(X_test_vectors)

    print("\nModel:", classifier.__class__.__name__)
    print(classification_report(y_test, predictions))

    # Display confusion matrix
    matrix = confusion_matrix(y_test, predictions)

    plt.figure()
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=["POLITICS", "SPORT"],
        yticklabels=["POLITICS", "SPORT"]
    )
    plt.title(classifier.__class__.__name__)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():

    print("Loading SPORT vs POLITICS dataset...\n")

    texts, labels = load_sport_politics_dataset()

    print("Total number of documents:", len(texts))

    # Split dataset into training and testing portions
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=42
    )

    # ------------------------------------------------------
    # Model 1: Bag of Words + Naive Bayes
    # ------------------------------------------------------
    print("\n=== Bag of Words + Multinomial Naive Bayes ===")

    train_and_evaluate_model(
        CountVectorizer(stop_words="english"),
        MultinomialNB(),
        train_texts,
        test_texts,
        train_labels,
        test_labels
    )

    # ------------------------------------------------------
    # Model 2: TF-IDF + Logistic Regression
    # ------------------------------------------------------
    print("\n=== TF-IDF + Logistic Regression ===")

    train_and_evaluate_model(
        TfidfVectorizer(stop_words="english"),
        LogisticRegression(max_iter=1000),
        train_texts,
        test_texts,
        train_labels,
        test_labels
    )

    # ------------------------------------------------------
    # Model 3: TF-IDF (Unigram + Bigram) + Linear SVM
    # ------------------------------------------------------
    print("\n=== TF-IDF (Unigram + Bigram) + Linear SVM ===")

    train_and_evaluate_model(
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
        LinearSVC(),
        train_texts,
        test_texts,
        train_labels,
        test_labels
    )

    # ======================================================
    # Interactive Prediction Using Best Model
    # ======================================================

    print("\nInteractive Mode (type 'exit' to quit)")

    best_feature_extractor = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    best_classifier = LinearSVC()

    # Train on full dataset for final deployment model
    full_dataset_vectors = best_feature_extractor.fit_transform(texts)
    best_classifier.fit(full_dataset_vectors, labels)

    while True:

        user_text = input("\nEnter a news article: ")

        if user_text.lower() == "exit":
            break

        user_vector = best_feature_extractor.transform([user_text])
        prediction = best_classifier.predict(user_vector)

        print("Predicted Category:", prediction[0])


if __name__ == "__main__":
    main()
