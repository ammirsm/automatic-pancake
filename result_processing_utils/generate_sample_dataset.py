import copy
import random

import pandas as pd

dataset_list = ["vande", "vandis", "cultural"]

models = ["LogisticRegression", "NaiveBayes", "SVM", "RandomForest"]
feature_extractors = ["TFIDF_Low", "TFIDF_High"]
features_before_and_after_list = ["baseline", "endnote", "fulltextpdf"]

strategies = ["max_prob", "uncertainty"]
label_column_list = ["title_label", "fulltext_label"]
filter_data_list = ["all", "endnote", "fulltext"]

csv = open("../data.csv", "wb")

number_of_points = {"cultural": 9339, "vande": 6184, "vandis": 10953}
relavant = {"cultural": 1200, "vande": 300, "vandis": 600}
the_id = 1
list_of_dicts = []
for dataset in dataset_list:
    line = {"dataset": dataset}
    for model in models:
        line["model"] = model
        for feature_extractor in feature_extractors:
            line["feature_extractor"] = feature_extractor
            for features_before_and_after in features_before_and_after_list:
                line["features_before_and_after"] = features_before_and_after
                for strategy in strategies:
                    line["strategy"] = strategy
                    for label_column in label_column_list:
                        line["label_column"] = label_column
                        for filter_data in filter_data_list:
                            line["filter_data"] = filter_data
                            line["id"] = the_id
                            y_axis = [
                                i for i in range(0, number_of_points[dataset], 50)
                            ]

                            x_axis = [
                                random.randint(0, relavant[dataset])
                                for _ in range(len(y_axis))
                            ]
                            x_axis_time = [
                                random.randint(0, 1000) for _ in range(len(y_axis))
                            ]
                            x_axis.sort()
                            x_axis_time.sort()

                            for y, x, x_time in zip(y_axis, x_axis, x_axis_time):
                                line_cache = copy.deepcopy(line)
                                line_cache["y_axis_founded"] = y
                                line_cache["y_axis_time"] = y
                                line_cache["x_axis_founded"] = x
                                line_cache["x_axis_time"] = x_time
                                line_cache["x_axis_founded_sem"] = round(
                                    random.uniform(0, 3), 2
                                )
                                line_cache["x_axis_time_sem"] = round(
                                    random.uniform(0, 3), 2
                                )
                                list_of_dicts.append(line_cache)

                            the_id += 1

print("done")


dataframe = pd.DataFrame(list_of_dicts)
dataframe.to_csv("data.csv", index=False)
