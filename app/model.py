import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import MultinomialNB


class LearningModel:
    def __init__(
        self,
        data,
        feature_before_vectorize,
        feature_after_vectorize,
        label_column="label",
        ngram_max=1,
        model=MultinomialNB(),
        the_percentile=100,
        number_of_sd=3,
        sampler=None,
        tokenizer="TF-IDF",
        revectorize=False,
        tokenizer_max_df=0.7,
        tokenizer_min_df=0.01,
    ):
        self.percentile = the_percentile
        self.label_column = label_column
        self.data = data.data.copy()
        self.ngram_max = ngram_max
        self.number_of_sd = number_of_sd
        self.sd_counter = 0
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.tokenizer_max_df = tokenizer_max_df
        self.tokenizer_min_df = tokenizer_min_df

        self.revectorize = revectorize
        self.features = feature_before_vectorize
        self.features_for_vectorize = feature_after_vectorize
        self._init_data()
        # print(f"number of paper is : {len(self.data)}")

        self.training_set = None
        self.test_set = None
        self.predicted_labels = None

        self.model = model

    def restart_model(self):
        self.data["training_set"] = 0

    def _init_data(self):
        self.data["features_vectorize"] = ""
        for i in self.features_for_vectorize:
            self.data["features_vectorize"] = (
                self.data["features_vectorize"] + " " + self.data[i]
            )

        self.data["features"] = ""
        for i in self.features:
            self.data["features"] = self.data["features"] + " " + self.data[i]

        self.data["training_set"] = 0
        self.data["proba_history"] = [[] for i in range(self.data.shape[0])]

        self.vectorize_init()

    def generate_matrix(self):

        # test_set, training_set = self.vectorize()
        test_set, training_set = self.get_set()
        self.label_set = self.data.loc[self.data.training_set == 1]["label"]
        self.feature_selection(test_set, training_set)

    def feature_selection(self, test_set, training_set):
        selector = SelectPercentile(f_classif, percentile=self.percentile)
        # selector = SelectKBest(chi2, k=self.data.shape[1] - 2)
        selector.fit(training_set, self.label_set)
        if self.tokenizer in ["sbert", "scibert"]:
            self.training_set = selector.transform(training_set)
            self.test_set = selector.transform(test_set)
        else:
            self.training_set = selector.transform(training_set).toarray()
            self.test_set = selector.transform(test_set).toarray()

    def balance_data(self):
        balance_data_type = self.sampler
        if not balance_data_type:
            return
        elif balance_data_type == "RandomOverSampler":
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(random_state=42)
        elif balance_data_type == "SMOTE":
            from imblearn.over_sampling import SMOTE

            sampler = SMOTE(random_state=42)
        elif balance_data_type == "ADASYN":
            from imblearn.over_sampling import ADASYN

            sampler = ADASYN(random_state=42)
        elif balance_data_type == "SMOTEENN":
            from imblearn.over_sampling import SMOTEENN

            sampler = SMOTEENN(random_state=42)
        elif balance_data_type == "SMOTETomek":
            from imblearn.over_sampling import SMOTETomek

            sampler = SMOTETomek(random_state=42)
        elif balance_data_type == "RandomUnderSampler":
            from imblearn.under_sampling import RandomUnderSampler

            sampler = RandomUnderSampler(random_state=42)
        elif balance_data_type == "NearMiss":
            from imblearn.under_sampling import NearMiss

            sampler = NearMiss(random_state=42)
        elif balance_data_type == "CondensedNearestNeighbour":
            from imblearn.under_sampling import CondensedNearestNeighbour

            sampler = CondensedNearestNeighbour(random_state=42)
        elif balance_data_type == "EditedNearestNeighbours":
            from imblearn.under_sampling import EditedNearestNeighbours

            sampler = EditedNearestNeighbours(random_state=42)
        self.training_set, self.label_set = sampler.fit_resample(
            self.training_set, self.label_set
        )

    def vectorize_init(self):
        tokenizer = self.tokenizer
        if tokenizer == "TF-IDF":
            self.tfidf()
        elif tokenizer == "sbert":
            self.sbert()
        elif tokenizer == "scibert":
            self.scibert()

    def tfidf(self):
        vectorizer = TfidfVectorizer(
            max_df=self.tokenizer_max_df,
            min_df=self.tokenizer_min_df,
            # max_features=1000,
            stop_words="english",
            ngram_range=(1, self.ngram_max),
        )
        # it should run in each iteration because we're changing vectorization in each iteration
        self.features_vectorized = vectorizer.fit_transform(
            self.data["features_vectorize"]
        )
        self.test_set_vectorized = vectorizer.transform(self.data["features_vectorize"])

    def get_set(self):
        row_index = self.data["training_set"].astype("bool")
        return self.test_set_vectorized, (self.features_vectorized)[row_index]

    def train_model(self):
        self.model.fit(self.training_set, self.label_set)
        predicted_labels = self.model.predict_proba(self.test_set)
        if 0 in self.data.columns:
            self.data = self.data.drop(columns=[0])
        if 1 in self.data.columns:
            self.data = self.data.drop(columns=[1])
        self.data = pd.concat([pd.DataFrame(predicted_labels), self.data], axis=1)

        self.data["proba_history"] += pd.DataFrame(
            [[[i]] for i in self.data[1].values.tolist()]
        )[0]

        self.data["sd_history_" + str(self.sd_counter)] = self.data[
            "proba_history"
        ].apply(lambda x: np.std(x))

        self.sd_counter += 1

    def scibert(self):
        self.features_vectorized = np.concatenate(
            (
                self.scibert_helper(self.data["title"]),
                self.scibert_helper(self.data["abstract"]),
            ),
            axis=1,
        )
        self.test_set_vectorized = self.features_vectorized.copy()

    def scibert_helper(self, data):
        import torch
        from transformers import AutoModel, AutoTokenizer

        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        # Load AutoModel from huggingface model repository
        tokenizer = AutoTokenizer.from_pretrained("gsarti/scibert-nli")
        model = AutoModel.from_pretrained("gsarti/scibert-nli")

        # Tokenize sentences
        data = list(data)
        encoded_input = tokenizer(
            data, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        return np.array(mean_pooling(model_output, encoded_input["attention_mask"]))
