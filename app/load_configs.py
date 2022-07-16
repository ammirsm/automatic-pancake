import copy
import sys

from app.configs import (
    cycle,
    data_set_path_list,
    feature_configs,
    feature_extractors,
    features_before_and_after,
    features_columns_cleaning,
    filter_data_list,
    label_column_list,
    models,
    number_of_papers,
    strategies,
)
from app.feature_extractor.mapper import get_feature_extractor_class

new_feature_configs = {}
for key, feature_config in feature_configs.items():
    for key_model, model in models.items():
        new_key = key + "_" + key_model
        new_feature_configs[new_key] = copy.deepcopy(feature_config)
        new_feature_configs[new_key]["model"] = model
feature_configs = copy.deepcopy(new_feature_configs)

new_feature_configs = {}
for key, feature_config in feature_configs.items():
    for key_ext, extractor in feature_extractors.items():
        new_key = key + "_" + key_ext
        new_feature_configs[new_key] = copy.deepcopy(feature_config)
        new_feature_configs[new_key]["extractor_key"] = key_ext
        new_feature_configs[new_key] = {**new_feature_configs[new_key], **extractor}
feature_configs = copy.deepcopy(new_feature_configs)

new_feature_configs = {}
for key, feature_config in feature_configs.items():
    for key_ext, extractor in features_before_and_after.items():
        new_key = key + "_" + key_ext
        new_feature_configs[new_key] = copy.deepcopy(feature_config)
        new_feature_configs[new_key]["features_key"] = key_ext
        new_feature_configs[new_key] = {**new_feature_configs[new_key], **extractor}

feature_configs = copy.deepcopy(new_feature_configs)

for key, feature_config in feature_configs.items():
    FeatureExtractorClass = get_feature_extractor_class(feature_config.pop("tokenizer"))
    feature_extractor = FeatureExtractorClass(
        feature_before_vectorize=feature_config.get("feature_before_vectorize"),
        feature_after_vectorize=feature_config.get("feature_after_vectorize"),
        the_percentile=feature_config.get("percentile"),
        # ngram_max=feature_config.pop('ngram_max'),
        tokenizer_max_df=feature_config.get("tokenizer_max_df"),
        tokenizer_min_df=feature_config.get("tokenizer_min_df"),
    )
    feature_configs[key]["feature_extractor"] = feature_extractor


for key, strategy in strategies.items():
    strategies[key] = {
        "configs": None,
        "step": 1,
    }

full_configs = []

for label_column in label_column_list:
    for filter_data in filter_data_list:
        for data_set_name, data_set_path in data_set_path_list.items():
            for strategy_name, strategy in strategies.items():
                for feature_config_name, feature_config in feature_configs.items():
                    full_configs.append(
                        {
                            "data": dict(
                                pickle_file=data_set_path,
                                label_column=label_column,
                                features_columns_cleaning=features_columns_cleaning,
                                filter_data=filter_data,
                                cycle=cycle,
                                papers_count=number_of_papers,
                            ),
                            "strategy": copy.deepcopy(strategy),
                            "strategy_name": strategy_name,
                            "data_set_name": data_set_name,
                            "feature_config": copy.deepcopy(feature_config),
                            "feature_config_name": feature_config_name,
                        }
                    )

main_directory_name = sys.argv[1] if len(sys.argv) >= 2 else None
