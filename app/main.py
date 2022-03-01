import timeit

from app.agent import ActiveLearningAgent
from app.configs import cycle, model_configs, number_of_iterations, strategies
from app.model import LearningModel

strategies_result = {}

for strategy in strategies:
    list_of_models_config = []
    for i in range(number_of_iterations):

        # for data_set in data_config:
        #     data_set._shuffle_data()

        the_percentile_title = 100
        the_percentile_fulltext = 20

        for key, config in model_configs.items():
            config["number_of_relevant"] = (
                config["data"].data.loc[config["data"].data.label == 1].shape[0]
            )
            if len(config["feature_columns"]) == 3:
                model_configs[key]["percentile"] = the_percentile_fulltext
            elif len(config["feature_columns"]) == 2:
                model_configs[key]["percentile"] = the_percentile_title
            model_configs[key]["learning_model"] = LearningModel(
                config["data"],
                feature_columns=config["feature_columns"],
                model=config["model"],
                the_percentile=config["percentile"],
            )
            model_configs[key]["agent"] = ActiveLearningAgent(
                learning_model=config["learning_model"],
                name=key,
                each_cycle=cycle,
                update_training_set_strategy=strategy[0],
                query_ratio=strategy[1],
            )
            print("******************************")
            print(key)
            print("******************************")
            start = timeit.default_timer()
            config["agent"].start_active_learning()
            stop = timeit.default_timer()
            print("Time: ", stop - start)
            model_configs[key]["plot_data"] = model_configs[key]["agent"].plot_data()
            model_configs[key]["learning_model"] = None
            model_configs[key]["agent"] = None

        list_of_models_config.append(model_configs)
    strategies_result[
        str(strategy[0]) + " - " + str(strategy[1])
    ] = list_of_models_config
