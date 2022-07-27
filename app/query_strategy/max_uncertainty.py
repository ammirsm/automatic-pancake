from .base import QueryStrategyBase


class Uncertainty(QueryStrategyBase):
    def update_training_set_strategy(self, max_uncertainty_number=None):
        max_uncertainty_number = (
            self.each_cycle if not max_uncertainty_number else max_uncertainty_number
        )
        if "uncertainty" in self.learning_model.data.columns:
            self.learning_model.data = self.learning_model.data.drop(
                columns=["uncertainty"]
            )
        self.learning_model.data["uncertainty"] = 1 - abs(
            self.learning_model.data[0] - 0.5
        )
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.training_set == 0)]
            .sort_values(by=["uncertainty"])
            .head(max_uncertainty_number)
            .index,
            "training_set",
        ] = 1
