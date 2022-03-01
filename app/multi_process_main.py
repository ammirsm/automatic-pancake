import multiprocessing
import timeit
import warnings

from agent import ActiveLearningAgent
from configs import cycle, model_configs, number_of_iterations, strategies
from import_export import export_data
from model import LearningModel

warnings.filterwarnings("ignore")

# from concurrent import futures


def start_active_learning(key, config, strategies_result, strategy_key):
    the_percentile_title = 100
    the_percentile_fulltext = 20
    config["number_of_relevant"] = (
        config["data"].data.loc[config["data"].data.label == 1].shape[0]
    )
    if len(config["feature_columns"]) == 3:
        config["percentile"] = the_percentile_fulltext
    elif len(config["feature_columns"]) == 2:
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
        update_training_set_strategy=strategy[0],
        query_ratio=strategy[1],
    )
    config["agent"].start_active_learning()
    config["plot_data"] = model_configs[key]["agent"].plot_data()
    config["learning_model"] = None
    config["agent"] = None
    strategies_result[strategy_key].append(model_configs)


strategies_result = {}

first = timeit.default_timer()
counter = 0
processes = []
# with futures.ThreadPoolExecutor(max_workers=15) as executor:
for strategy in strategies:
    strategy_key = str(strategy[0]) + " - " + str(strategy[1])
    strategies_result[strategy_key] = []
    for i in range(number_of_iterations):

        # for data_set in data_config:
        #     data_set._shuffle_data()

        # the_percentile_title = 100
        # the_percentile_fulltext = 20

        for key, config in model_configs.items():
            print(counter)
            p = multiprocessing.Process(
                target=start_active_learning,
                args=(key, config, strategies_result, strategy_key),
            )
            processes.append(p)
            p.start()

            # executor.submit(start_active_learning, key, config)
            counter += 1

for p in processes:
    p.join()

second = timeit.default_timer()
print("Total Time : ", second - first)
export_data(strategies_result, "asghar2.pickle")
