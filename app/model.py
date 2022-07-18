import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from app.sampler.mapper import get_sampler_class


class LearningModel:
    def __init__(
        self,
        data,
        feature_extractor,
        label_column="label",
        model=MultinomialNB(),
        number_of_sd=3,
        revectorize=False,
        sampler=None,
    ):
        self.label_column = label_column
        self.data = data.data.copy()
        self.number_of_sd = number_of_sd
        self.sd_counter = 0
        self.SamplerClass = get_sampler_class(sampler)
        self.feature_extractor = feature_extractor

        self._init_data()
        # print(f"number of paper is : {len(self.data)}")

        self.revectorize = revectorize

        self.training_set = None
        self.test_set = None
        self.predicted_labels = None

        self.model = model

    def restart_model(self):
        self.data["training_set"] = 0

    def _init_data(self):
        self.feature_extractor.init_vectorizer_data(self.data)

        self.data["training_set"] = 0
        self.data["proba_history"] = [[] for i in range(self.data.shape[0])]

        self.feature_extractor.vectorize_init()

    def generate_matrix(self):

        # test_set, training_set = self.vectorize()
        training_set_row_index = self.data["training_set"].astype("bool")

        self.label_set = self.data.loc[self.data.training_set == 1]["label"]
        training_set, test_set = self.feature_extractor.feature_selection(
            training_set_row_index, self.label_set
        )
        self.training_set = training_set
        self.test_set = test_set

    def balance_data(self):
        if not self.SamplerClass:
            return
        sampler = self.SamplerClass()
        self.training_set, self.label_set = sampler.fit_resample(
            self.training_set, self.label_set
        )

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
