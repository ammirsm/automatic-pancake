from max_prob import MaxProb
from max_uncertainty import Uncertainty

from .base import QueryStrategyBase


class MixChecking(QueryStrategyBase):
    def __init__(self, query_ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_prob_obj = MaxProb(*args, **kwargs)
        self.uncertainty_obj = Uncertainty(*args, **kwargs)
        self.query_ratio = query_ratio

    def update_training_set_strategy(self):
        number_of_training_set = self.learning_model.data[
            (self.learning_model.data.training_set == 0)
        ].shape[0]
        active_learning_cycle = number_of_training_set / self.each_cycle
        # query ratio is the number which max prob is running
        if active_learning_cycle % (self.query_ratio + 1):
            self.uncertainty_obj.update_training_set_strategy()
        else:
            self.max_prob_obj.update_training_set_strategy()
