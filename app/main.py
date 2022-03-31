import os
import timeit
import warnings
from datetime import datetime

from agent import ActiveLearningAgent
from configs import cycle, model_configs, number_of_iterations, result, strategies
from import_export import export_data
from model import LearningModel

from app.draw_utils import draw_helper

warnings.filterwarnings("ignore")


def start_active_learning(key, config, strategies_result, strategy_key, dirs):
    the_percentile_title = 100
    the_percentile_fulltext = 100
    config["number_of_relevant"] = (
        config["data"].data.loc[config["data"].data.label == 1].shape[0]
    )
    if len(config["feature_columns"]) == 3:
        config["percentile"] = the_percentile_fulltext
    elif len(config["feature_columns"]) == 2:
        config["percentile"] = the_percentile_title
    else:
        config["percentile"] = the_percentile_title

    config["learning_model"] = LearningModel(
        config["data"],
        feature_columns=config["feature_columns"],
        model=config["model"],
        the_percentile=config["percentile"],
    )

    config["agent"] = ActiveLearningAgent(
        learning_model=config["learning_model"],
        name=key,
        each_cycle=cycle,
        update_training_set_strategy=strategy_key.split(" - ")[0],
        query_ratio=int(strategy_key.split(" - ")[1]),
    )
    config["agent"].start_active_learning()
    config["plot_data"] = config["agent"].plot_data()

    path = result + "/".join(dirs)
    os.makedirs(path, exist_ok=True)
    export_data(config, f"{path}/{key}.pickle")
    config["learning_model"] = None
    config["agent"] = None
    strategies_result[strategy_key].append(model_configs)
    return path


if __name__ == "__main__":
    strategies_result = {}

    first = timeit.default_timer()
    counter = 0
    processes = []
    dirs = [None] * 3

    now = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")  # current date and time
    dirs[0] = now
    for strategy in strategies:
        strategy_key = str(strategy[0]) + " - " + str(strategy[1])
        dirs[1] = strategy_key
        strategies_result[strategy_key] = []
        for i in range(number_of_iterations):
            dirs[2] = str(i)
            for key, config in model_configs.items():
                print(counter)
                path = start_active_learning(
                    key, config, strategies_result, strategy_key, dirs.copy()
                )
                counter += 1

    second = timeit.default_timer()
    draw_helper(os.path.join(result, dirs[0]))
    print("Total Time : ", second - first)
