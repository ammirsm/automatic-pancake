import os

from matplotlib import pyplot as plt

from app.configs import output_dir, result
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
    first = True
    y_axis = []
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
        if first:
            y_axis = [model_config["plot_data"][0], model_config["number_of_relevant"]]
            first = False
    return y_axis


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


def generate_strategy_data(strategy_data, path, y_axis):
    list_of_strategy = os.listdir(path)
    list_of_strategy.remove("output")
    for strategy in list_of_strategy:
        strategy_path = os.path.join(path, strategy)
        list_of_iterations = os.listdir(strategy_path)
        iteration_data = {}
        for iteration in list_of_iterations:
            iteration_path = os.path.join(strategy_path, iteration)
            if y_axis:
                iteration_plot_data(iteration_path, iteration_data)
            else:
                y_axis = iteration_plot_data(iteration_path, iteration_data)
        strategy_data[strategy] = iteration_data
    return y_axis


def draw_helper(average_result, main_directory_name):
    for dataset_name, dataset in average_result.items():
        for strategy_name, strategy in dataset["strategies"].items():
            dirs = [
                main_directory_name,
                dataset_name,
                "average",
                strategy_name,
            ]
            path = result + "/".join(dirs)
            os.makedirs(path, exist_ok=True)

            plt.rcParams["figure.figsize"] = [15, 9]
            plt.rcParams["figure.dpi"] = 100
            plt.title(f"{dataset_name} {strategy_name}")
            plt.ylabel("% of found relavant papers")
            plt.xlabel("# of reviewed papers")
            for config_name, config in strategy["configs"].items():
                plt.plot(dataset["data"].y_axis, config["plot_data"], label=config_name)
            output_file_path = os.path.join(path, f"{strategy_name}.pdf")
            plt.legend()
            plt.savefig(output_file_path)
            plt.clf()  # clear figure
    #
    # if not strategies_result_pickle_file_path[-1] == "/":
    #     strategies_result_pickle_file_path += "/"
    # output_path = os.path.join(strategies_result_pickle_file_path, output_dir)
    # os.makedirs(output_path, exist_ok=True)
    # strategy_data = {}
    # y_axis = []
    # y_axis = generate_strategy_data(
    #     strategy_data, strategies_result_pickle_file_path, y_axis
    # )
    # # number_of_relevant = y_axis[1]
    # y_axis = y_axis[0]
    # strategy_mean_data = {}
    # generate_mean_plot_data(strategy_mean_data, strategy_data)
    # for strategy_k, strategy_d in strategy_mean_data.items():
