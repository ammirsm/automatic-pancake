from .base import QueryStrategyBase


class MaxProb(QueryStrategyBase):
    def update_training_set_strategy(self, max_prob_number=None):
        max_prob_number = self.each_cycle if not max_prob_number else max_prob_number
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.training_set == 0)]
            .sort_values(by=[0])
            .head(self.each_cycle)
            .index,
            "training_set",
        ] = 1
