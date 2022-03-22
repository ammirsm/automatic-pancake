import os

from matplotlib import pyplot as plt

from app.configs import output_dir
from app.draw import strategies_result_pickle_file_path
from app.import_export import import_data


def draw_model(model_config):
    label = model_config["agent"].name
    x = model_config["plot_data"][0]
    y = model_config["plot_data"][1]
    save_plot(label, model_config, x, y)


def save_plot(label, model_config, x, y):
    plt.rcParams["figure.figsize"] = [15, 9]
    plt.rcParams["figure.dpi"] = 100
    plt.title(model_config["agent"].name)
    plt.ylabel("% of found relavant papers")
    plt.xlabel("# of reviewed papers")
    plt.plot(x, y, label=label)
    output_file_path = os.path.join(output_dir, f"{label}.pdf")
    plt.savefig(output_file_path)


def iteration_plot_data(the_iteration_path, the_iteration_data):
    for model in os.listdir(the_iteration_path):
        model_path = os.path.join(the_iteration_path, model)
        model_config = import_data(model_path)
        if not model_config["agent"].name in the_iteration_data.keys():
            the_iteration_data[model_config["agent"].name] = [
                model_config["plot_data"][1]
            ]
        else:
            the_iteration_data[model_config["agent"].name].append(
                model_config["plot_data"][1]
            )
    return the_iteration_data


def mean(_list):
    return sum(_list) / len(_list)


def average_list_helper(list_of_lists):
    the_new_list = []
    for i, data in enumerate(list_of_lists[0]):
        get_average_list = []
        for j, list_ in enumerate(list_of_lists):
            get_average_list.append(list_[i])
        the_new_list.append(mean(get_average_list))
    return the_new_list


def generate_mean_plot_data(strategy_mean_data, strategy_data):
    for strategy_k, strategy_d in strategy_data.items():
        strategy_mean_data[strategy_k] = {}
        for iteration_k, iteration_d in strategy_d.items():
            strategy_mean_data[strategy_k][iteration_k] = average_list_helper(
                iteration_d
            )


def generate_strategy_data(strategy_data):
    list_of_strategy = os.listdir(strategies_result_pickle_file_path)
    for strategy in list_of_strategy:
        strategy_path = os.path.join(strategies_result_pickle_file_path, strategy)
        list_of_iterations = os.listdir(strategy_path)
        iteration_data = {}
        for iteration in list_of_iterations:
            iteration_path = os.path.join(strategy_path, iteration)
            iteration_plot_data(iteration_path, iteration_data)
        strategy_data[strategy] = iteration_data
