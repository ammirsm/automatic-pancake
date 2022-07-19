import copy
import os
import uuid
import warnings
from datetime import datetime

from app.agent import ActiveLearningAgent
from app.configs import cycle, number_of_iterations, result
from app.data import Data
from app.import_export import export_json
from app.load_configs import full_configs, main_directory_name
from app.model import LearningModel

warnings.filterwarnings("ignore")


def export_json_file(dirs, file_name, the_data):
    path = result + "/".join(dirs)
    os.makedirs(path, exist_ok=True)
    export_json(the_data, f"{path}/{file_name}.json")


if __name__ == "__main__":
    if not main_directory_name:
        main_directory_name = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    first_time = datetime.now()
    print("START", flush=True)
    for the_config in full_configs:
        for iteration in range(number_of_iterations):
            tic = datetime.now()
            cache_object = copy.deepcopy(the_config)
            the_config["data_obj"] = Data(**the_config["data"])
            the_config["data_obj"].init_data()
            if len(the_config["data_obj"].label_unique_values()) != 2:
                the_config["data_obj"] = None
                print("Not binary classification")
                continue
            name = f'{the_config["data_set_name"]} - {the_config["strategy_name"]} - {the_config["feature_config_name"]}'
            print(
                "-------\n",
                f"{name} - {the_config['data']['label_column']} - {the_config['data']['filter_data']} ",
                flush=True,
            )
            features = the_config["feature_config"]
            the_config["learning_model"] = LearningModel(
                the_config["data_obj"],
                model=features["model"],
                sampler=features["sampler"],
                revectorize=features["revectorize"],
                feature_extractor=features["feature_extractor"],
            )
            QueryStrategyClass = the_config.pop("strategy_class")
            query_strategy = QueryStrategyClass(
                learning_model=the_config["learning_model"],
                each_cycle=cycle,
                query_ratio=the_config["strategy"]["step"],
                prioritize=features["prioritize"],
            )
            the_config["agent"] = ActiveLearningAgent(
                query_strategy=query_strategy,
                learning_model=the_config["learning_model"],
                name=name,
            )
            the_config["agent"].start_active_learning()

            plot_data = the_config["agent"].plot_data()

            the_config["plot_data"] = plot_data
            the_config["times"] = the_config["agent"].times_spent
            the_config["number_of_papers"] = the_config["data_obj"].number_of_papers
            the_config["number_of_relavant"] = the_config["data_obj"].number_of_relavant
            the_config["learning_model"] = None
            the_config["data_obj"] = None
            the_config["agent"] = None
            the_config["feature_config"]["feature_extractor"] = None
            features["model"] = str(features["model"])

            export_json_file(
                [
                    main_directory_name,
                    the_config["data_set_name"],
                    the_config["data"]["label_column"],
                    the_config["data"]["filter_data"],
                    the_config["strategy_name"],
                    str(features["model"])[:-2],
                    str(features["extractor_key"]),
                    str(features["features_key"]),
                ],
                str(uuid.uuid4().hex),
                the_config,
            )
            the_config = copy.deepcopy(cache_object)
            print("time: ", datetime.now() - tic, flush=True)

    print("-------\ntotal time spend: ", datetime.now() - first_time)
    print(
        "-----------------------------------------END----------------------------------------",
        flush=True,
    )
