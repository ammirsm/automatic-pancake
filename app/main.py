import os
import warnings
from datetime import datetime

from agent import ActiveLearningAgent
from configs import cycle, full_configs, number_of_iterations, result
from import_export import export_data
from model import LearningModel

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
    config_pointer["plot_data"] = config["agent"].plot_data()


if __name__ == "__main__":
    main_directory_name = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")

    for dataset_name, dataset in full_configs.items():
        for iteration in range(number_of_iterations):
            dataset["data"].init_data()
            for strategy_name, strategy in dataset["strategies"].items():
                for config_name, config in strategy["configs"].items():
                    name = f"{dataset_name} - {config_name} - {strategy_name} - {strategy['step']}"
                    config_pointer = full_configs[dataset_name]["strategies"][
                        strategy_name
                    ]["configs"][config_name]

                    run(config, config_pointer)

                    export_config(
                        [
                            main_directory_name,
                            dataset_name,
                            str(iteration),
                            strategy_name,
                        ],
                        config_name,
                    )

                    config_pointer["learning_model"] = None  # to save memory
                    config_pointer["agent"] = None  # to save memory

                    print(f"{name} is done")
