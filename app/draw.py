import os

import matplotlib.pyplot as plt
import numpy

from app.configs import output_dir

is_exist = os.path.exists(output_dir)
if not is_exist:
    # Create a new directory because it does not exist
    os.makedirs(output_dir)


def draw(list_of_models_config, title):
    model_configs = list_of_models_config[0]

    def mean(_list):
        return sum(_list) / len(_list)

    def sum_list(list_1, list_2):
        for i, data in enumerate(list_1):
            list_1[i] = data + list_2[i]

    plt.rcParams["figure.figsize"] = [15, 9]
    plt.rcParams["figure.dpi"] = 100
    plt.title(title)
    plt.ylabel("% of found relavant papers")
    plt.xlabel("# of reviewed papers")

    for key, config in model_configs.items():
        total_papers_count, founded_papers_percentile = [
            0 for i in range(len(config["plot_data"][0]))
        ], [0 for i in range(len(config["plot_data"][1]))]
        for the_model_config in list_of_models_config:
            sum_list(total_papers_count, the_model_config[key]["plot_data"][0])
            sum_list(founded_papers_percentile, the_model_config[key]["plot_data"][1])
        total_papers_count, founded_papers_percentile = numpy.array(
            total_papers_count
        ) / len(list_of_models_config), numpy.array(founded_papers_percentile) / len(
            list_of_models_config
        )
        plt.plot(total_papers_count, founded_papers_percentile, label=key)

    title_number_of_relevant = []
    fulltext_number_of_relevant = []
    for the_model_config in list_of_models_config:
        title_number_of_relevant.append(
            the_model_config["title_lr"]["number_of_relevant"]
        )
        fulltext_number_of_relevant.append(
            model_configs["fulltext_lr"]["number_of_relevant"]
        )

    plt.plot(
        [0, mean(title_number_of_relevant)],
        [0, 100],
        label="best_line_title",
        linestyle="dashed",
        color="black",
    )

    plt.plot(
        [0, mean(fulltext_number_of_relevant)],
        [0, 100],
        label="best_line_fulltext",
        linestyle="dashed",
        color="gray",
    )

    plt.plot(
        total_papers_count,
        [i / total_papers_count[-1] * 100 for i in total_papers_count],
        label="base_line",
        linestyle="dashed",
        color="black",
    )

    plt.legend()

    output_file_path = os.path.join(output_dir, f"{title}.pdf")
    plt.savefig(output_file_path)


if __name__ == "__main__":
    from sys import argv

    from import_export import import_data

    strategies_result_pickle_file_path = argv[1]
    strategies_result = import_data(strategies_result_pickle_file_path)
    # execute only if run as a script
    for key in strategies_result.keys():
        draw(strategies_result[key], key)
