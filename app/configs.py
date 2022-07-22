class Config:
    def __init__(
        self,
        data,
        strategy,
        strategy_name,
        strategy_class,
        data_set_name,
        feature_config,
        feature_config_name,
        cycle,
        number_of_iterations,
    ):
        self.data = data
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.strategy_class = strategy_class
        self.data_set_name = data_set_name
        self.feature_config = feature_config
        self.feature_config_name = feature_config_name
        self.number_of_iterations = number_of_iterations
        self.cycle = cycle

    @property
    def name(self):
        return (
            f"{self.data_set_name} - {self.strategy_name} - {self.feature_config_name}"
        )

    @property
    def key(self):
        return (
            f"{self.name} - {self.data['label_column']} - {self.data['filter_data']} "
        )

    @property
    def feature_extractor_data(self):
        return {
            "feature_before_vectorize": self.feature_config.get(
                "feature_before_vectorize"
            ),
            "feature_after_vectorize": self.feature_config.get(
                "feature_after_vectorize"
            ),
            "the_percentile": self.feature_config.get("percentile"),
            # "ngram_max": self.feature_config.pop('ngram_max'),
            "tokenizer_max_df": self.feature_config.get("tokenizer_max_df"),
            "tokenizer_min_df": self.feature_config.get("tokenizer_min_df"),
        }

    @property
    def learning_model_data(self):
        return {
            "model": self.feature_config["model"],
            "sampler": self.feature_config["sampler"],
            "revectorize": self.feature_config["revectorize"],
        }

    @property
    def query_strategy_data(self):
        return {
            "each_cycle": self.cycle,
            "query_ratio": self.strategy["step"],
            "prioritize": self.feature_config["prioritize"],
        }

    @property
    def agent_data(self):
        return {"name": self.name}

    @property
    def export_data(self):
        feature_config = self.feature_config.copy()
        feature_config.pop("feature_extractor_class")
        feature_config["model"] = str(feature_config["model"])

        return {
            "data": self.data,
            "strategy": self.strategy,
            "strategy_name": self.strategy_name,
            "data_set_name": self.data_set_name,
            "feature_config": feature_config,
        }

    @property
    def output_dir(self):
        return [
            self.data_set_name,
            self.data["label_column"],
            self.data["filter_data"],
            self.strategy_name,
            str(self.feature_config["model"])[:-2],
            str(self.feature_config["extractor_key"]),
            str(self.feature_config["features_key"]),
        ]
