from max_prob import MaxProb
from max_uncertainty import Uncertainty

from .base import QueryStrategyBase


class MixCheckingInside(QueryStrategyBase):
    def __init__(self, query_ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_prob_obj = MaxProb(*args, **kwargs)
        self.uncertainty_obj = Uncertainty(*args, **kwargs)
        self.query_ratio = query_ratio

    def update_training_set_strategy(self):
        max_uncertainty_number = int(self.each_cycle / (self.query_ratio + 1))
        max_prob_number = int(self.each_cycle - max_uncertainty_number)
        self.max_prob_obj.update_training_set_strategy(max_uncertainty_number)
        self.uncertainty_obj.update_training_set_strategy(max_prob_number)
