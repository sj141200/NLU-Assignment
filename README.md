Sports vs Politics Text Classification

Roll Number: M25CSE031

Project Overview

This project implements a machine learning-based text classification system that automatically classifies news documents into one of two categories:

SPORT

POLITICS

The system compares multiple feature extraction techniques and machine learning algorithms to determine which model performs best for this classification task.

Objectives

The main goals of this project are:

Collect and prepare a real-world dataset.

Convert raw text into numerical feature representations.

Train and evaluate at least three different machine learning models.

Compare performance using quantitative metrics.

Provide an interactive prediction mode for new text input.

Dataset
Source

The dataset used is the 20 Newsgroups Dataset, accessed via scikit-learn.

Selected Categories

SPORT

rec.sport.baseball

rec.sport.hockey

POLITICS

talk.politics.misc

talk.politics.guns

talk.politics.mideast

Preprocessing

The dataset is loaded with:

Headers removed

Footers removed

Quotes removed

The final dataset contains several thousand documents split into:

70% Training Data

30% Testing Data

Feature Engineering Techniques

Three feature representation methods were used:

Bag of Words (BoW)

Converts text into word frequency vectors.

Simple and effective baseline representation.

TF-IDF (Term Frequency – Inverse Document Frequency)

Weighs words based on importance.

Reduces impact of common words.

TF-IDF with Bigrams

Includes both single words (unigrams) and two-word combinations (bigrams).

Captures contextual phrases such as:

“world cup”

“prime minister”

Machine Learning Models Used

Three classifiers were trained and compared:

Model	Feature Type	Description
Multinomial Naive Bayes	Bag of Words	Probabilistic text classifier
Logistic Regression	TF-IDF	Linear classifier with strong baseline performance
Linear SVM	TF-IDF (1,2)	Margin-based classifier, best performance
Evaluation Metrics

Each model was evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Observed Results (Example)
Model	Feature Representation	Accuracy
Naive Bayes	Bag of Words	~85–88%
Logistic Regression	TF-IDF	~90–92%
Linear SVM	TF-IDF + Bigrams	~93–95%

Best Performing Model:
Linear SVM with TF-IDF (Unigrams + Bigrams)

How to Run the Project
Step 1: Install Required Libraries
pip install scikit-learn matplotlib seaborn

Step 2: Run the Python Script
python M25CSE031_prob4.py

Step 3: View Model Performance

The program will:

Train all three models

Print classification reports

Display confusion matrices

Step 4: Interactive Prediction

After evaluation, the program enters interactive mode:

Enter a news article:


Type any paragraph related to sports or politics, and the model will predict:

Predicted Category: SPORT


or

Predicted Category: POLITICS


Type exit to stop.

Project Structure
Project Folder
 ├── M25CSE031_prob4.py
 ├── README.md
 └── Report.pdf

Limitations

Vocabulary overlap between domains can cause confusion.

Linear models do not understand deep semantic meaning.

No hyperparameter tuning was performed.

Deep learning models (e.g., BERT) may outperform this system.

Future Improvements

Apply hyperparameter optimization.

Use word embeddings (Word2Vec, GloVe).

Implement deep learning models.

Expand dataset to more domains.

Conclusion

This project demonstrates how classical machine learning techniques can effectively solve real-world text classification problems.

Key Findings:

Feature engineering significantly affects performance.

TF-IDF performs better than simple Bag of Words.

Linear SVM achieved the highest accuracy.

Bigram features improve contextual understanding.
