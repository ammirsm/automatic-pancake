import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


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
        self.sampler = sampler
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
