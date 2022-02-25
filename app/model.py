import timeit

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import pandas as pd
import numpy as np


class LearningModel:
    def __init__(self, data, feature_columns, label_column='label', ngram_max=1,
                 model=MultinomialNB(), the_percentile=50, number_of_sd=3):
        self.percentile = the_percentile
        self.features = feature_columns
        self.label_column = label_column
        self.data = data.data.copy()
        self.ngram_max = ngram_max
        self.number_of_sd = number_of_sd
        self.sd_counter = 0

        self._init_data()
        print(f"number of paper is : {len(self.data)}")

        self.training_set = None
        self.test_set = None
        self.predicted_labels = None

        self.model = model

    def restart_model(self):
        self.data['training_set'] = 0

    def _init_data(self):
        if self.label_column != 'label':
            self.data['label'] = self.data[self.label_column]
        self.data['features'] = ''
        for f in self.features:
            self.data['features'] += self.data[f]
        self.data['training_set'] = 0
        self.data['proba_history'] = [[] for i in range(self.data.shape[0])]

        self.vectorize_init()

    def generate_matrix(self):
        print(len(self.data.loc[self.data.training_set == 1]['features']))
        start = timeit.default_timer()
        # test_set, training_set = self.vectorize()
        test_set, training_set = self.get_set()
        stop = timeit.default_timer()
        print('Time: vectorize ', stop - start)
        start = timeit.default_timer()
        self.label_set = self.data.loc[self.data.training_set == 1]['label']
        self.feature_selection(test_set, training_set)
        stop = timeit.default_timer()
        print('Time: feature_selection ', stop - start)
        print(self.training_set.shape)

    def feature_selection(self, test_set, training_set):
        selector = SelectPercentile(f_classif, percentile=self.percentile)
        selector.fit(training_set, self.label_set)
        self.training_set = selector.transform(training_set).toarray()
        self.test_set = selector.transform(test_set).toarray()

    def vectorize_init(self):

        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english', ngram_range=(1, self.ngram_max))

        self.features_vectorized = vectorizer.fit_transform(self.data['features'])

        self.test_set_vectorized = vectorizer.transform(self.data['features'])

    def get_set(self):
        row_index = self.data['training_set'].astype('bool')
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

        self.data['proba_history'] += pd.DataFrame([[[i]] for i in self.data[0].values.tolist()])[0]

        # for i in range(self.number_of_sd):
        #   if i == 0:
        #     continue
        #   self.data['sd_history_'+str(i - 1)] = self.data['sd_history_'+str(i)]

        self.data['sd_history_' + str(self.sd_counter)] = self.data['proba_history'].apply(lambda x: np.std(x))
        # delete_header = 'sd_history_' + str(self.sd_counter - self.number_of_sd)
        # if delete_header in list(self.data.columns):
        #   del self.data[delete_header]

        self.sd_counter += 1
        # for i in range(self.data.shape[0]):
        #   self.data['sd_history_'+str(self.sd_counter)].iloc[i] = np.std(self.data['proba_history'].iloc[i])
