import copy
import json
from glob import glob

import numpy as np
import pandas as pd

path = "./server_result/**/*.json"

json_files = glob(path, recursive=True)


def load_json_to_dict(json_file):
    with open(json_file) as f:
        return json.load(f)


files_list = [load_json_to_dict(file) for file in json_files]
files_df = pd.DataFrame(files_list)

list_of_dict_data = []
identifier = 0
for i, row in files_df.iterrows():
    # this_group_file = json_files[i : i + 10]
    group_df = files_df[files_df.strategy_name == row["strategy_name"]][
        files_df.data_set_name == row["data_set_name"]
    ][files_df.feature_config_name == row["feature_config_name"]][
        files_df.data == row["data"]
    ]
    json_data = group_df.to_dict("records")
    if not json_data:
        continue
    # json_data = [load_json_to_dict(file) for file in this_group_file]
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
df.to_csv("./server_result/preprocess_parallel_result.csv", index=False)

# https://drive.google.com/drive/u/0/folders/1Ob0AM1ElCen5ATf5yZ3iO41KsUSaPua7
# https://public.tableau.com/app/profile/ranjan.bhattarai/viz/AmirTableauDashboard/Dashboard?publish=yes
