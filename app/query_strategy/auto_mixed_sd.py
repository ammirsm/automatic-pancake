from max_prob import MaxProb
from max_uncertainty import Uncertainty

from .base import QueryStrategyBase


class AutoMixedSD(QueryStrategyBase):
    def __init__(self, query_ratio=1, sd_threshold=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_prob_obj = MaxProb(*args, **kwargs)
        self.uncertainty_obj = Uncertainty(*args, **kwargs)
        self.query_ratio = query_ratio
        self.sd_threshold = sd_threshold

    def check_sd_threshold(self):
        if self.learning_model.sd_counter < 25:
            return False
        data_zero_training_set = self.learning_model.data.loc[
            self.learning_model.data.training_set == 0
        ]
        t = 0
        flag = True
        for i in range(
            self.learning_model.sd_counter - self.learning_model.number_of_sd,
            self.learning_model.sd_counter,
        ):
            t += data_zero_training_set.loc[
                data_zero_training_set["sd_history_" + str(i)] <= self.sd_threshold
            ].shape[0]
            if not (
                data_zero_training_set.loc[data_zero_training_set.training_set == 0][
                    "sd_history_" + str(i)
                ]
                <= self.sd_threshold
            ).all():
                flag = False
        return flag

    def update_training_set_strategy(self):
        if self.check_sd_threshold():
            self.max_prob_obj.update_training_set_strategy()
        else:
            self.uncertainty_obj.update_training_set_strategy()
