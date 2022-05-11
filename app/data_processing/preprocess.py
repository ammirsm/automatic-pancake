from app.import_export import import_data


class PreprocessBase:
    def __init__(self, data_path):
        self.data = import_data(data_path)
        pass


class PreprocessTFIDF(PreprocessBase):
    def process(self):
        pass
