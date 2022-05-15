import os
import warnings
from datetime import datetime

import numpy as np
from agent import ActiveLearningAgent
from configs import cycle, full_configs, number_of_iterations, result
from import_export import export_data
from model import LearningModel

from app.draw_utils import draw_helper

warnings.filterwarnings("ignore")


def export_config(dirs, name):
    path = result + "/".join(dirs)
    os.makedirs(path, exist_ok=True)
    export_data(config, f"{path}/{name}.pickle")


def run(config, config_pointer):
    config_pointer["learning_model"] = LearningModel(
        dataset["data"],
        model=config["model"],
        the_percentile=config["percentile"],
        sampler=config["sampler"],
        tokenizer=config["tokenizer"],
        revectorize=config["revectorize"],
    )
    config_pointer["agent"] = ActiveLearningAgent(
        learning_model=config["learning_model"],
        name=name,
        each_cycle=cycle,
        update_training_set_strategy=strategy_name,
        query_ratio=strategy["step"],
        prioritize=config["prioritize"],
    )

    config["agent"].start_active_learning()

    return config["agent"].plot_data()


if __name__ == "__main__":
    main_directory_name = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")

    aggregate_results = []
    full_configs_cache = full_configs.copy()
    for iteration in range(number_of_iterations):
        full_configs = full_configs_cache.copy()
        for dataset_name, dataset in full_configs.items():
            dataset["data"].init_data()
            for strategy_name, strategy in dataset["strategies"].items():
                for config_name, config in strategy["configs"].items():
                    name = f"{dataset_name} - {strategy_name} - {config_name} - {strategy['step']}"
                    print(config)
                    config["learning_model"] = LearningModel(
                        dataset["data"],
                        model=config["model"],
                        the_percentile=config["percentile"],
                        sampler=config["sampler"],
                        tokenizer=config["tokenizer"],
                        revectorize=config["revectorize"],
                    )
                    print(config)
                    config["agent"] = ActiveLearningAgent(
                        learning_model=config["learning_model"],
                        name=name,
                        each_cycle=cycle,
                        update_training_set_strategy=strategy_name,
                        query_ratio=strategy["step"],
                        prioritize=config["prioritize"],
                    )

                    config["agent"].start_active_learning()

                    plot_data = config["agent"].plot_data()

                    config["plot_data"] = plot_data

                    export_config(
                        [
                            main_directory_name,
                            dataset_name,
                            str(iteration),
                            strategy_name,
                        ],
                        config_name,
                    )

                    config["learning_model"] = None

                    config["agent"] = None

                    full_configs[dataset_name]["strategies"][strategy_name]["configs"][
                        config_name
                    ] = config

                    print(f"{name} is done")
        aggregate_results.append(full_configs)
    # mean of all results
    if number_of_iterations > 1:
        average_result = full_configs

    for dataset_name, dataset in average_result.items():
        for strategy_name, strategy in dataset["strategies"].items():
            for config_name, config in strategy["configs"].items():
                print(
                    f"{dataset_name} - {config_name} - {strategy_name} - {strategy['step']}"
                )

                plot_data_matrix = []
                average_result_pointer = average_result[dataset_name]["strategies"][
                    strategy_name
                ]["configs"][config_name]

                for iteration in range(number_of_iterations):
                    result_pointer = aggregate_results[iteration][dataset_name][
                        "strategies"
                    ][strategy_name]["configs"][config_name]
                    print(result_pointer["plot_data"][1])
                    plot_data_matrix.append(result_pointer["plot_data"][1])
                plot_data_matrix = np.matrix(plot_data_matrix)
                average_result_pointer["plot_data"] = np.array(
                    plot_data_matrix.mean(0)
                ).tolist()[0]

    draw_helper(aggregate_results)
