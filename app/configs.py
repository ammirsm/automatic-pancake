from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from data import Data

strategies = [("max_prob", 1), ("uncertainty", 1), ("mix_checking_inside", 1), ("mix_checking_inside", 2),
              ("mix_checking_inside", 3), ("mix_checking_inside", 4), ("mix_checking_inside", 5)]

number_of_papers = 1000
number_of_iterations = 1
sd_threshold = 0.1
cycle = 10

title_data = Data('./asset/Cultural/dataset_keywords.csv', papers_count=number_of_papers)
fulltext_data = Data('./asset/Cultural/dataset_keywords.csv', label_csv_file='./asset/Cultural/fulltext_label.csv', papers_count=number_of_papers)
model_configs = {
    "title_lr": {
        "data": title_data,
        "feature_columns": ['title', 'abstract'],
        'model': LogisticRegression(),
    },
    "title_lr_keywords": {
        "data": title_data,
        "feature_columns": ['title', 'abstract', 'keywords'],
        'model': LogisticRegression(),
    },
    "title_nb": {
        "data": title_data,
        "feature_columns": ['title', 'abstract'],
        'model': MultinomialNB(),
    },
    "title_nb_keywords": {
        "data": title_data,
        "feature_columns": ['title', 'abstract', 'keywords'],
        'model': MultinomialNB(),
    },
    "fulltext_lr": {
        "data": fulltext_data,
        "feature_columns": ['title', 'abstract'],
        'model': LogisticRegression(),
    },
    "fulltext_lr_keywords": {
        "data": fulltext_data,
        "feature_columns": ['title', 'abstract', 'keywords'],
        'model': LogisticRegression(),
    },
    "fulltext_nb": {
        "data": fulltext_data,
        "feature_columns": ['title', 'abstract'],
        'model': MultinomialNB(),
    },
    "fulltext_nb_keywords": {
        "data": fulltext_data,
        "feature_columns": ['title', 'abstract', 'keywords'],
        'model': MultinomialNB(),
    },
}
