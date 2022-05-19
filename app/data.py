import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.utils import shuffle

from app.import_export import import_data


class Data:
    def __init__(
        self,
        pickle_file,
        label_column,
        features_columns_cleaning,
        filter_data="all",
        papers_count=None,
        cycle=20,
    ):
        self.filter_data = filter_data
        self.pickle_file = pickle_file
        self.cycle = cycle
        self.papers_count = papers_count
        self.label_column = label_column
        self.features_columns = features_columns_cleaning
        self.pure_data = import_data(self.pickle_file)
        self.init_data()
        self.number_of_papers = self.data.shape[0]

    def init_data(self):
        self.data = pd.DataFrame(self.pure_data)
        self.data = self.data.fillna("")
        self.data["label"] = self.data[self.label_column]
        self.number_of_relavant = self.data[self.data["label"] == 1].shape[0]
        self._clean_data()
        self._shuffle_data()
        self._cut_data()
        self.update_profile()
        if self.filter_data == "all":
            pass
        elif self.filter_data == "endnote":
            self.data = self.data[self.data["endnote-pdf_text"] != ""]
        elif self.filter_data == "fulltext":
            self.data = self.data[self.data[" "] != ""]

    def update_profile(self):
        self.profile = ProfileReport(self.data, title="Data Profile Report")

    def _clean_data(self):
        self.data.dropna(subset=self.features_columns, inplace=True)

    def _shuffle_data(self):
        self.data = shuffle(self.data)
        self.data.reset_index(inplace=True, drop=True)

    def _cut_data(self):
        if self.papers_count and self.papers_count < len(self.data):
            self.data = self.data[: self.papers_count]

    def restart_data(self):
        self.data.drop("training_set")
        self._shuffle_data()
