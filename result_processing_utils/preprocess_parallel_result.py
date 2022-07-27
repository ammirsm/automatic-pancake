import copy
import json
from glob import glob

import numpy as np
import pandas as pd

path = "./app/result/server_result 3/**/*.json"

json_files = glob(path, recursive=True)


def load_json_to_dict(json_file):
    with open(json_file) as f:
        return json.load(f)


files_list = []
for file in json_files:
    json_file_dict = load_json_to_dict(file)
    json_file_dict["path"] = file
    file_splited = file.split("/")
    json_file_dict["file_directory"] = "/".join(file_splited[7:-1])
    files_list.append(json_file_dict)

files_df = pd.DataFrame(files_list)

list_of_dict_data = []
identifier = 0

list_of_conditions = []

for i, row in files_df.iterrows():
    group_df = files_df[files_df.file_directory == row["file_directory"]]
    json_data = group_df.to_dict("records")
    if not json_data:
        continue

    list_of_conditions.append(
        dict(
            strategy_name=row["strategy_name"],  # 2
            data_set_name=row["data_set_name"],  # 3
            feature_extraction=row["feature_config"]["extractor_key"],  # 3
            model=row["feature_config"]["model"],  # 4
            features_key=row["feature_config"]["features_key"],  # 2
            label=row["data"]["label_column"],  # 2 + 1
            filter=row["data"]["filter_data"],  # 2 + 1
        )
    )

    files_df.drop(group_df.index, axis=0, inplace=True)
    plot_data = []
    times = []
    y_axis = json_data[0]["plot_data"][0]
    for data in json_data:
        plot_data.append(data["plot_data"][1])
        times.append(data["times"])
    plot_data = np.matrix(plot_data)
    times = np.matrix(times)
    plot_data_sd = np.array(np.std(plot_data, axis=0)).tolist()
    times_sd = np.array(np.std(times, axis=0)).tolist()
    plot_data = np.array(plot_data.mean(0)).tolist()
    times = np.array(times.mean(0)).tolist()
    plot_data = plot_data[0]
    plot_data_sd = plot_data_sd[0]
    times = times[0]
    times_sd = times_sd[0]

    for each, data in enumerate(plot_data):
        plot_data[each] = plot_data[each] / 100 * json_data[0]["number_of_relavant"]

    for time_i, time in enumerate(times):
        if time_i in [0, 1]:
            continue
        times[time_i] = times[time_i] - times[time_i - 1]

    the_json_data_sample = json_data[0]
    dict_data = {
        "dataset": the_json_data_sample["data_set_name"],
        "model": the_json_data_sample["feature_config"]["model"][:-2],
        "number_of_relavant": the_json_data_sample["number_of_relavant"],
        "number_of_papers": the_json_data_sample["number_of_papers"],
        "feature_extractor": the_json_data_sample["feature_config"]["extractor_key"],
        "strategy": the_json_data_sample["strategy_name"],
        "label_column": the_json_data_sample["data"]["label_column"],
        "filter_data": the_json_data_sample["data"]["filter_data"],
        "features_before_and_after": the_json_data_sample["feature_config"][
            "features_key"
        ],
    }

    for reviewed_papers, relevant_papers, time_of_train, time_sd, relevant_sd in zip(
        y_axis, plot_data, times, times_sd, plot_data_sd
    ):
        line_cache = copy.deepcopy(dict_data)
        line_cache["x_axis_founded"] = reviewed_papers
        line_cache["x_axis_time"] = reviewed_papers
        line_cache["y_axis_founded"] = relevant_papers
        line_cache["y_axis_time"] = time_of_train
        line_cache["y_axis_founded_sem"] = relevant_sd
        line_cache["y_axis_time_sem"] = time_sd
        line_cache["id"] = identifier
        list_of_dict_data.append(line_cache)

    identifier += 1

df = pd.DataFrame(list_of_dict_data)

# https://drive.google.com/drive/u/0/folders/1Ob0AM1ElCen5ATf5yZ3iO41KsUSaPua7
# https://public.tableau.com/app/profile/ranjan.bhattarai/viz/AmirTableauDashboard/Dashboard?publish=yes


dataset_list = ["vande", "vandis", "cultural"]

models = [
    "SVC(probability=True)",
    "MultinomialNB()",
    "RandomForestClassifier()",
    "LogisticRegression()",
]
feature_extractors = ["TFIDF_High", "TFIDF_Low", "BagOfWords"]
features_key = ["baseline", "endnote", "fulltextpdf"]

strategies = ["max_prob", "uncertainty"]
label_column_list_1 = ["title_label", "fulltext_label"]
filter_data_list_1 = ["all", "endnote"]

label_column_list_2 = ["fulltext_label"]
filter_data_list_2 = ["fulltext"]


list_of_combinations = []
for dataset in dataset_list:
    for model in models:
        for feature_extractor in feature_extractors:
            for features_before_and_after in features_key:
                for strategy in strategies:
                    for label_column in label_column_list_1:
                        for filter_data in filter_data_list_1:
                            list_of_combinations.append(
                                dict(
                                    data_set_name=dataset,
                                    model=model,
                                    feature_extraction=feature_extractor,
                                    features_key=features_before_and_after,
                                    strategy_name=strategy,
                                    label=label_column,
                                    filter=filter_data,
                                )
                            )
                    for label_column in label_column_list_2:
                        for filter_data in filter_data_list_2:
                            list_of_combinations.append(
                                dict(
                                    data_set_name=dataset,
                                    model=model,
                                    feature_extraction=feature_extractor,
                                    features_key=features_before_and_after,
                                    strategy_name=strategy,
                                    label=label_column,
                                    filter=filter_data,
                                )
                            )


df_combinations = pd.DataFrame(list_of_combinations)
df_conditions = pd.DataFrame(list_of_conditions)

# remove df conditions from df combinations
for index, row in df_conditions.iterrows():
    a = df_combinations[
        (df_combinations["data_set_name"] == row["data_set_name"])
        & (df_combinations["model"] == row["model"])
        & (df_combinations["feature_extraction"] == row["feature_extraction"])
        & (df_combinations["features_key"] == row["features_key"])
        & (df_combinations["strategy_name"] == row["strategy_name"])
        & (df_combinations["label"] == row["label"])
        & (df_combinations["filter"] == row["filter"])
    ]
    df_combinations = df_combinations.drop(a.index)


df.to_csv("./app/result/server_result 3/preprocess_paresult.csv", index=False)
