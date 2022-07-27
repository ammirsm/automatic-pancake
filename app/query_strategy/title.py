from .base import QueryStrategyBase


class Title(QueryStrategyBase):
    def update_training_set_strategy(self):
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.training_set == 0)]
            .sort_values(by=["title"])
            .head(10)
            .index,
            "training_set",
        ] = 1
