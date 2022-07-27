import copy
import json
import sys
from datetime import datetime
from itertools import product

from app.feature_extraction.utils import get_feature_extraction_class
from app.query_strategy.utils import get_query_class
from app.utils import get_model_class


class Config:
    main_directory_name = (
        sys.argv[1]
        if len(sys.argv) >= 2
        else datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    )
    result = "results/"

    def __init__(
        self,
        data,
        strategy,
        feature_config,
        number_of_iterations,
    ):
        self.data = data
        self.strategy = strategy
        self.feature_config = feature_config
        self.number_of_iterations = number_of_iterations

    @property
    def name(self):
        return f"{self.data['name']} - {self.strategy.get('name')} - {self.feature_config.get('name')}"

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
            "each_cycle": self.data.get("cycle"),
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

        strategy_config = self.strategy.copy()
        strategy_config.pop("strategy_class")
        return {
            "data": self.data,
            "strategy": strategy_config,
            "strategy_name": self.strategy["name"],
            "data_set_name": self.data["name"],
            "feature_config": feature_config,
        }

    @property
    def output_dir(self):
        return [
            self.data["name"],
            self.data["label_column"],
            self.data["filter_data"],
            self.strategy.get("name"),
            self.feature_config["name"],
        ]


def load_from_json(filepath):
    with open(filepath, "r") as file:
        configs_data = json.load(file)
    result = configs_data.get("result")
    if result:
        Config.result = configs_data.get("result")

    def _get_feature_configs_list():
        # return list of all combinations of feature_configs
        models = configs_data.get("models")
        feature_configs = configs_data.get("feature_configs")
        feature_extractors = configs_data.get("feature_extractors")
        features_before_and_after = configs_data.get("features_before_and_after")
        feature_configs_combinations = product(
            models.items(),
            feature_configs.items(),
            features_before_and_after.items(),
            feature_extractors.items(),
        )
        feature_configs_list = []
        for (
            model,
            feature_config,
            feature_before_after,
            feature_extractor,
        ) in feature_configs_combinations:
            new_feature_config = {
                "name": f"{model[0]} - {feature_config[0]} - {feature_before_after[0]} - {feature_extractor[0]}"
            }

            # Fill model data
            model_data = model[1]
            ModelClass = get_model_class(model_data.get("name"))
            model = (
                ModelClass(**model_data.get("kwargs"))
                if model_data.get("kwargs")
                else ModelClass()
            )
            new_feature_config["model"] = model

            # Fill feature extractor data
            feature_extractor_data = feature_extractor[1]
            FeatureExtractorClass = get_feature_extraction_class(
                feature_extractor_data.get("tokenizer")
            )
            new_feature_config["feature_extractor_class"] = FeatureExtractorClass
            new_feature_config.update(feature_extractor_data)

            # Fill feature before and featre after
            new_feature_config.update(feature_before_after[1])

            # Fill feature configs data
            new_feature_config.update(feature_config[1])

            feature_configs_list.append(new_feature_config)
        return feature_configs_list

    feature_configs_list = _get_feature_configs_list()

    full_configs = product(
        configs_data.get("data_set_path_list").items(),
        configs_data.get("label_column_list"),
        configs_data.get("filter_data_list"),
        configs_data.get("strategies"),
        feature_configs_list,
    )

    configs = []
    for (
        (data_set_name, data_set_path),
        label_column,
        filter_data,
        strategy,
        feature_config,
    ) in full_configs:
        configs.append(
            Config(
                data=dict(
                    data_set_file=data_set_path,
                    name=data_set_name,
                    label_column=label_column,
                    features_columns_cleaning=configs_data.get(
                        "features_columns_cleaning"
                    ),
                    filter_data=filter_data,
                    cycle=configs_data.get("cycle"),
                    papers_count=configs_data.get("number_of_papers"),
                ),
                strategy={
                    "name": strategy,
                    "configs": None,
                    "step": 1,
                    "strategy_class": get_query_class(strategy),
                },
                feature_config=copy.deepcopy(feature_config),
                number_of_iterations=configs_data.get("number_of_iterations"),
            )
        )
    return configs
