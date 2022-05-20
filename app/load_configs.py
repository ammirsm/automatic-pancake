import copy

from data import Data

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
        new_feature_configs[new_key] = {**new_feature_configs[new_key], **extractor}
feature_configs = copy.deepcopy(new_feature_configs)

new_feature_configs = {}
for key, feature_config in feature_configs.items():
    for key_ext, extractor in features_before_and_after.items():
        new_key = key + "_" + key_ext
        new_feature_configs[new_key] = copy.deepcopy(feature_config)
        new_feature_configs[new_key] = {**new_feature_configs[new_key], **extractor}
feature_configs = copy.deepcopy(new_feature_configs)

for key, strategy in strategies.items():
    strategies[key] = {
        "configs": copy.deepcopy(feature_configs),
        "step": 1,
    }

full_configs = {}

for label_column in label_column_list:
    for filter_data in filter_data_list:
        for data_set_name, data_set_path in data_set_path_list.items():
            full_configs[label_column + "_" + filter_data + "_" + data_set_name] = {
                "data": Data(
                    pickle_file=data_set_path,
                    label_column=label_column,
                    features_columns_cleaning=features_columns_cleaning,
                    filter_data=filter_data,
                    cycle=cycle,
                    papers_count=number_of_papers,
                ),
                "strategies": copy.deepcopy(strategies),
            }
