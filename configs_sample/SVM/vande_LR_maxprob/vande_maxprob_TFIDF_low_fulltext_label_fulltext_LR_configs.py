# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

number_of_papers = None
number_of_iterations = 10
cycle = 50
features_columns_cleaning = [
    "title",
    "abstract",
    "processed_title",
    "processed_abstract",
]
output_dir = "output"
result = "result/"
models = {
    # "LogisticRegression": LogisticRegression(),
    # "NaiveBayes": MultinomialNB(),
    "SVM": SVC(probability=True),
    # "RandomForest": RandomForestClassifier(n_estimators=100),
}
feature_extractors = {
    "TFIDF_Low": {
        "tokenizer": "TF-IDF",
        "tokenizer_max_df": 0.7,
        "tokenizer_min_df": 0.2,
    },
    # "TFIDF_High": {
    #     "tokenizer": "TF-IDF",
    #     "tokenizer_max_df": 0.9,
    #     "tokenizer_min_df": 0.1,
    # },
    # "BagOfWords": {
    #     "tokenizer": "BOW",
    # }
}
features_before_and_after = {
    "baseline": {
        "feature_before_vectorize": ["title", "abstract"],
        "feature_after_vectorize": ["title", "abstract"],
        "revectorize": False,
    },
    "endnote": {
        "feature_before_vectorize": [
            "processed_title",
            "processed_abstract",
            "processed_endnote_pdf_text",
        ],
        "feature_after_vectorize": [
            "processed_title",
            "processed_abstract",
            "processed_endnote_pdf_text",
        ],
        "revectorize": False,
    },
    "fulltextpdf": {
        "feature_before_vectorize": [
            "processed_title",
            "processed_abstract",
            "processed_endnote_pdf_text",
        ],
        "feature_after_vectorize": [
            "processed_title",
            "processed_abstract",
            "processed_endnote_pdf_text",
            "processed_pdf_manual_pdf_text",
        ],
        "revectorize": True,
    },
}
strategies = {
    "max_prob": None,
    # "uncertainty": None,
}
label_column_list = [
    # "title_label",
    "fulltext_label"
]
filter_data_list = [
    # "all",
    # "endnote",
    "fulltext"
]
data_set_path_list = {
    "vande": "./asset/pickle_datasets/vande_full.pickle",
}
feature_configs = {
    "-": {
        "sampler": None,
        "percentile": 100,
        "prioritize": False,
    },
}
