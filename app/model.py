import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import MultinomialNB


class LearningModel:
    def __init__(
        self,
        data,
        feature_columns,
        label_column="label",
        ngram_max=1,
        model=MultinomialNB(),
        the_percentile=100,
        number_of_sd=3,
        sampler=None,
        tokenizer="TF-IDF",
        revectorize=False,
        features_for_vectorize=None,
    ):
        self.percentile = the_percentile
        self.features = feature_columns
        self.label_column = label_column
        self.data = data.data.copy()
        self.ngram_max = ngram_max
        self.number_of_sd = number_of_sd
        self.sd_counter = 0
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.revectorize = revectorize
        self.features_for_vectorize = features_for_vectorize
        self._init_data()
        # print(f"number of paper is : {len(self.data)}")

        self.training_set = None
        self.test_set = None
        self.predicted_labels = None

        self.model = model

    def restart_model(self):
        self.data["training_set"] = 0

    def _init_data(self):
        if self.label_column != "label":
            self.data["label"] = self.data[self.label_column]

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

    def tfidf(self):
        vectorizer = TfidfVectorizer(
            max_df=0.7,
            min_df=0.01,
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

    # def vectorize(self):
    #     start = timeit.default_timer()

    #     vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
    #                                  stop_words='english', ngram_range=(1, self.ngram_max))
    #     stop = timeit.default_timer()
    #     print('Time: TfidfVectorizer ', stop - start)
    #     start = timeit.default_timer()

    #     training_set = vectorizer.fit_transform(self.data.loc[self.data.training_set == 1]['features'])
    #     stop = timeit.default_timer()
    #     print('Time: vectorizer.fit_transform ', stop - start)
    #     start = timeit.default_timer()

    #     test_set = vectorizer.transform(self.data['features'])
    #     stop = timeit.default_timer()
    #     print('Time: vectorizer.transform ', stop - start)

    #     return test_set, training_set

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

        # for i in range(self.number_of_sd):
        #   if i == 0:
        #     continue
        #   self.data['sd_history_'+str(i - 1)] = self.data['sd_history_'+str(i)]

        self.data["sd_history_" + str(self.sd_counter)] = self.data[
            "proba_history"
        ].apply(lambda x: np.std(x))
        # delete_header = 'sd_history_' + str(self.sd_counter - self.number_of_sd)
        # if delete_header in list(self.data.columns):
        #   del self.data[delete_header]

        self.sd_counter += 1
        # for i in range(self.data.shape[0]):
        #   self.data['sd_history_'+str(self.sd_counter)].iloc[i] = np.std(self.data['proba_history'].iloc[i])

    def sbert(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("../asset/model/scibert-nli")
        # we haven't tokenized the sentences yet, so we need to do it
        self.features_vectorized = model.encode(self.data["features_vectorize"])
        self.test_set_vectorized = model.encode(self.data["features_vectorize"])
