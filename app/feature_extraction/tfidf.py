from sklearn.feature_extraction.text import TfidfVectorizer

from .base import FeatureExtractorBase


class TFidf(FeatureExtractorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(tokenizer="TF-IDF", *args, **kwargs)

    def vectorize_init(self):
        vectorizer = TfidfVectorizer(
            max_df=self.tokenizer_max_df,
            min_df=self.tokenizer_min_df,
            max_features=5000,
            stop_words="english",
            ngram_range=(1, self.ngram_max),
        )
        # it should run in each iteration because we're changing vectorization in each iteration
        self.features_vectorized = vectorizer.fit_transform(self.feature_vectorize_data)
        self.test_set_vectorized = vectorizer.transform(self.feature_vectorize_data)

    def feature_selection(self, training_set_row_index, label_set):
        training_set, test_set = super().feature_selection(
            training_set_row_index, label_set
        )
        return training_set.toarray(), test_set.toarray()
