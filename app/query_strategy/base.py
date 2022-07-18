from abc import abstractmethod


class QueryStrategyBase:
    def __init__(self, learning_model, each_cycle=10):
        self.learning_model = learning_model
        self.features_vectorize_key = (
            self.learning_model.feature_extractor.features_vectorize_key
        )
        self.features_key = self.learning_model.feature_extractor.features_key
        self.each_cycle = each_cycle

    def update_training_set_features(self):
        self.learning_model.data.loc[
            self.learning_model.data.training_set == 1, self.features_vectorize_key
        ] = self.learning_model.data.loc[
            self.learning_model.data.training_set == 1, self.features_key
        ].values

    def update_training_set(self):
        self.update_training_set_strategy()
        self.update_training_set_features()

    @abstractmethod
    def update_training_set_strategy(self):
        pass
