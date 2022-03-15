# import multiprocessing
import timeit
import warnings
from concurrent import futures
from datetime import datetime

from configs import model_configs, number_of_iterations, strategies
from import_export import export_data
from main import start_active_learning

warnings.filterwarnings("ignore")


strategies_result = {}

first = timeit.default_timer()
counter = 0
processes = []
threads = []
dirs = [None] * 3

now = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")  # current date and time
dirs[0] = now
with futures.ThreadPoolExecutor(max_workers=15) as executor:
    for strategy in strategies:
        strategy_key = str(strategy[0]) + " - " + str(strategy[1])
        dirs[1] = strategy_key
        strategies_result[strategy_key] = []
        for i in range(number_of_iterations):
            dirs[2] = str(i)
            # for data_set in data_config:
            #     data_set._shuffle_data()

            # the_percentile_title = 100
            # the_percentile_fulltext = 20

            for key, config in model_configs.items():
                print(counter)
                # p = multiprocessing.Process(
                #     target=start_active_learning,
                #     args=(key, config, strategies_result, strategy_key),
                # )
                # processes.append(p)
                # p.start()
                threads.append(
                    executor.submit(
                        start_active_learning,
                        key,
                        config.copy(),
                        strategies_result,
                        strategy_key,
                        dirs.copy(),
                    )
                )
                counter += 1

# for p in processes:
#     p.join()

second = timeit.default_timer()
print("Total Time : ", second - first)
export_data(strategies_result, "asghar2.pickle")
