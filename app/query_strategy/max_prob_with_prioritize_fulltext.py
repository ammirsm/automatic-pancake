from .base import QueryStrategyBase


class MaxProbPrioritizeFulltext(QueryStrategyBase):
    def __init__(self, prioritize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prioritize = prioritize

    def update_training_set_strategy(self, max_prob_number=None):
        if (
            self.prioritize
            and self.learning_model.data[
                (self.learning_model.data.training_set == 0)
                & (self.learning_model.data.keywords_new != "")
            ].shape[0]
            > 0
        ):
            self.learning_model.data.loc[
                self.learning_model.data[
                    (self.learning_model.data.training_set == 0)
                    & (self.learning_model.data.keywords_new != "")
                ]
                .sort_values(by=[0])
                .head(self.each_cycle)
                .index,
                "training_set",
            ] = 1
        else:
            self.learning_model.data.loc[
                self.learning_model.data[(self.learning_model.data.training_set == 0)]
                .sort_values(by=[0])
                .head(self.each_cycle)
                .index,
                "training_set",
            ] = 1
