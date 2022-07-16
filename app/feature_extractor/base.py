from abc import abstractmethod

from sklearn.feature_selection import SelectPercentile, f_classif


class FeatureExtractorBase:
    def __init__(
        self,
        tokenizer,
        feature_before_vectorize,
        feature_after_vectorize,
        the_percentile=100,
        ngram_max=1,
        tokenizer_max_df=0.7,
        tokenizer_min_df=0.01,
    ):
        self.percentile = the_percentile
        self.tokenizer = tokenizer
        self.tokenizer_max_df = tokenizer_max_df
        self.tokenizer_min_df = tokenizer_min_df
        self.ngram_max = ngram_max

        self.features = feature_before_vectorize
        self.features_for_vectorize = feature_after_vectorize

        self.features_vectorized = None
        self.test_set_vectorized = None

    def init_vectorizer_data(self, data):
        data["features_vectorize"] = ""
        for i in self.features_for_vectorize:
            data["features_vectorize"] = data["features_vectorize"] + " " + data[i]

        data["features"] = ""
        for i in self.features:
            data["features"] = data["features"] + " " + data[i]

        self.feature_vectorize_data = data["features_vectorize"]

    def feature_selection(self, training_set_row_index, label_set):
        training_set = (self.features_vectorized)[training_set_row_index]

        selector = SelectPercentile(f_classif, percentile=self.percentile)
        # selector = SelectKBest(chi2, k=self.data.shape[1] - 2)
        selector.fit(training_set, label_set)

        return selector.transform(training_set), selector.transform(
            self.test_set_vectorized
        )

    @abstractmethod
    def vectorize_init(self):
        pass
