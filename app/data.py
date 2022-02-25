from sklearn.utils import shuffle
import pandas as pd
from pandas_profiling import ProfileReport


class Data:
    def __init__(self, csv_file, papers_count=None, label_csv_file=None):
        self.papers_count = papers_count
        self.data = pd.read_csv(csv_file)
        if label_csv_file:
            self.data['label'] = pd.read_csv(label_csv_file)['label']
        self._clean_data()
        self._shuffle_data()
        self._cut_data()
        self.update_profile()

    def update_profile(self):
        self.profile = ProfileReport(self.data, title='Data Profile Report')

    def _clean_data(self):
        self.data = self.data.dropna()

    def _shuffle_data(self):
        self.data = shuffle(self.data)
        self.data.reset_index(inplace=True, drop=True)

    def _cut_data(self):
        if self.papers_count and self.papers_count < len(self.data):
            self.data = self.data[:self.papers_count]

    def restart_data(self):
        self.data.drop('training_set')
        self._shuffle_data()
