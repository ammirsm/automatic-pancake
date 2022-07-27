import json
import timeit

import matplotlib.pyplot as plt


class ActiveLearningAgent:
    def __init__(
        self,
        learning_model,
        query_strategy,
        name=None,
        each_cycle=10,
        query_ratio=1,
        sd_threshold=0.1,
        prioritize=False,
    ):
        self.name = name
        self.learning_model = learning_model
        self.init_prior_knowledge()
        self.query_strategy = query_strategy
        self.founded_papers_count = [0]
        self.total_papers_count = [0]
        self.vectorized_cycle = 0
        self.times_spent = [0.0]

    def init_prior_knowledge(self, positive_papers_count=1, negative_papers_count=1):
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.label == 1)]
            .head(positive_papers_count)
            .index,
            "training_set",
        ] = 1
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.label == 0)]
            .head(negative_papers_count)
            .index,
            "training_set",
        ] = 1

    def update_training_set(self):
        # call update training set function of query strategy object
        self.query_strategy.update_training_set()
        new_founded_papers_count = (
            self.founded_papers_count[-1]
            - self.founded_papers_count[self.vectorized_cycle]
        )

        if self.learning_model.revectorize and new_founded_papers_count > 10:
            print("vectorize again")
            self.vectorized_cycle = len(self.founded_papers_count) - 1
            self.learning_model.feature_extractor.vectorize_init()

    def start_active_learning(self):
        start_time = timeit.default_timer()
        functinon_time = {
            "name": self.name,
            "strategy": type(self.query_strategy).__name__,
            "generate_matrix": 0,
            "train_model": 0,
            "update_training_set": 0,
            "total": 0,
        }
        while (
            len(
                self.learning_model.data.loc[
                    self.learning_model.data["training_set"] == 0
                ]
            )
            > 9
        ):
            # print( 'generate_matrix', f'{self.name} - {self.update_training_set_strategy}')
            functinon_time["generate_matrix"]
            tic = timeit.default_timer()
            self.learning_model.generate_matrix()
            functinon_time["generate_matrix"] += timeit.default_timer() - tic

            # print('train_model', f'{self.name} - {self.update_training_set_strategy}')
            tic = timeit.default_timer()

            self.learning_model.balance_data()
            self.learning_model.train_model()
            functinon_time["train_model"] += timeit.default_timer() - tic

            # print('update_training_set', f'{self.name} - {self.update_training_set_strategy}')
            tic = timeit.default_timer()
            self.update_training_set()
            functinon_time["update_training_set"] += timeit.default_timer() - tic

            self.founded_papers_count.append(
                len(
                    self.learning_model.data.loc[
                        self.learning_model.data.training_set == 1
                    ].loc[self.learning_model.data.label == 1]
                ),
            )
            self.total_papers_count.append(
                len(
                    self.learning_model.data.loc[
                        self.learning_model.data.training_set == 1
                    ]
                ),
            )
            self.times_spent.append(round(timeit.default_timer() - start_time, 2))
        functinon_time["total"] += timeit.default_timer() - start_time

        # print('-------\n', self.name, 'time')
        print(json.dumps(functinon_time, indent=2), end="\n\n")

    def plot(self, **keywords):
        founded_papers_percentile = [
            i / self.founded_papers_count[-1] * 100 for i in self.founded_papers_count
        ]
        plt.plot(self.total_papers_count, founded_papers_percentile, **keywords)

    def plot_data(self):
        founded_papers_percentile = [
            i / self.founded_papers_count[-1] * 100 for i in self.founded_papers_count
        ]
        self.total_papers_count.insert(0, 0)
        founded_papers_percentile.insert(0, 0)
        return [self.total_papers_count, founded_papers_percentile]
