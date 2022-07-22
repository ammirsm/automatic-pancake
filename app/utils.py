from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def get_model_class(name):
    mapper = {
        "LogisticRegression": LogisticRegression,
        "NaiveBayes": MultinomialNB,
        "SVM": SVC,
        "RandomForest": RandomForestClassifier,
    }
    return mapper[name]
