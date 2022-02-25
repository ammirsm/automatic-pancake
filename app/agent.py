import matplotlib.pyplot as plt
import timeit


class ActiveLearningAgent:
    def __init__(self, learning_model, update_training_set_strategy='max_prob', name=None, each_cycle=10, query_ratio=1,
                 sd_threshold=0.1):
        self.query_ratio = query_ratio
        self.each_cycle = each_cycle
        self.name = name
        self.learning_model = learning_model
        self.sd_threshold = sd_threshold
        self.init_prior_knowledge()
        self.update_training_set_strategy = update_training_set_strategy
        self.founded_papers_count = []
        self.total_papers_count = []

    def init_prior_knowledge(self, positive_papers_count=5, negative_papers_count=5):
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.label == 1)].head(
                positive_papers_count).index, 'training_set'] = 1
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.label == 0)].head(
                negative_papers_count).index, 'training_set'] = 1

    def update_training_set(self):
        if self.update_training_set_strategy == 'max_prob':
            self.max_prob_strategy()
        elif self.update_training_set_strategy == 'title':
            self.sort_by_title_strategy()
        elif self.update_training_set_strategy == 'uncertainty':
            self.max_uncertainty()
        elif self.update_training_set_strategy == 'mix_checking':
            self.mix_checking()
        elif self.update_training_set_strategy == 'mix_checking_inside':
            self.mix_checking_inside()
        elif self.update_training_set_strategy == 'auto_mix_sd':
            self.auto_mix_sd()

        else:
            raise ValueError("update_training_set_strategy error")

    def auto_mix_sd(self):
        if self.check_sd_threshold():
            self.max_prob_strategy()
        else:
            self.max_uncertainty()

    def sort_by_title_strategy(self):
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.training_set == 0)].sort_values(by=["title"]).head(
                10).index, 'training_set'] = 1

    def mix_checking(self):
        number_of_training_set = self.learning_model.data[(self.learning_model.data.training_set == 0)].shape[0]
        active_learning_cycle = number_of_training_set / self.each_cycle
        # query ratio is the number which max prob is running
        if active_learning_cycle % (self.query_ratio + 1):
            self.max_uncertainty()
        else:
            self.max_prob_strategy()

    def mix_checking_inside(self):
        max_uncertainty_number = int(self.each_cycle / (self.query_ratio + 1))
        max_prob_number = int(self.each_cycle - max_uncertainty_number)
        self.max_uncertainty(max_uncertainty_number)
        self.max_prob_strategy(max_prob_number)

    def max_uncertainty(self, max_uncertainty_number=None):
        max_uncertainty_number = self.each_cycle if not max_uncertainty_number else max_uncertainty_number
        if "uncertainty" in self.learning_model.data.columns:
            self.learning_model.data = self.learning_model.data.drop(columns=["uncertainty"])
        self.learning_model.data["uncertainty"] = 1 - abs(self.learning_model.data[0] - 0.5)
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.training_set == 0)].sort_values(by=["uncertainty"]).head(
                max_uncertainty_number).index, 'training_set'] = 1

    def max_prob_strategy(self, max_prob_number=None):
        max_prob_number = self.each_cycle if not max_prob_number else max_prob_number
        self.learning_model.data.loc[
            self.learning_model.data[(self.learning_model.data.training_set == 0)].sort_values(by=[0]).head(
                self.each_cycle).index, 'training_set'] = 1

    def check_sd_threshold(self):
        if self.learning_model.sd_counter < 25:
            return False
        data_zero_training_set = self.learning_model.data.loc[self.learning_model.data.training_set == 0]
        t = 0
        flag = True
        for i in range(self.learning_model.sd_counter - self.learning_model.number_of_sd,
                       self.learning_model.sd_counter):
            t += data_zero_training_set.loc[data_zero_training_set['sd_history_' + str(i)] <= self.sd_threshold].shape[
                0]
            if not (data_zero_training_set.loc[data_zero_training_set.training_set == 0][
                        'sd_history_' + str(i)] <= self.sd_threshold).all():
                flag = False
        print('-----------------------------------check_sd_threshold---------------------------------------------')
        print('number of true : ', t)
        print('avg of number of true : ', t / 3)
        print('sd counter : ', self.learning_model.sd_counter)
        print('number of zero training_set : ', data_zero_training_set.shape[0])
        print('using max_prob : ', flag)
        print(data_zero_training_set[['sd_history_' + str(self.learning_model.sd_counter - 1),
                                      'sd_history_' + str(self.learning_model.sd_counter - 2),
                                      'sd_history_' + str(self.learning_model.sd_counter - 3)]].describe())
        print('-----------------------------------END_check_sd_threshold---------------------------------------------')
        return flag

    def start_active_learning(self):
        while len(self.learning_model.data.loc[self.learning_model.data['training_set'] == 0]) > 9:
            start = timeit.default_timer()
            self.learning_model.generate_matrix()
            stop = timeit.default_timer()
            print('Time: generate_matrix ', stop - start)
            start = timeit.default_timer()
            self.learning_model.train_model()
            stop = timeit.default_timer()
            print('Time: train_model ', stop - start)
            start = timeit.default_timer()
            self.update_training_set()
            stop = timeit.default_timer()
            print('Time: update_training_set ', stop - start)
            start = timeit.default_timer()
            self.founded_papers_count.append(
                len(self.learning_model.data.loc[self.learning_model.data.training_set == 1].loc[
                        self.learning_model.data.label == 1]),
            )
            stop = timeit.default_timer()
            print('Time: founded_papers_count ', stop - start)
            start = timeit.default_timer()
            self.total_papers_count.append(
                len(self.learning_model.data.loc[self.learning_model.data.training_set == 1]),
            )
            stop = timeit.default_timer()
            print('Time: total_papers_count ', stop - start, end='\n\n\n')

    def plot(self, **keywords):
        founded_papers_percentile = [i / self.founded_papers_count[-1] * 100 for i in self.founded_papers_count]
        plt.plot(self.total_papers_count, founded_papers_percentile, **keywords)

    def plot_data(self):
        founded_papers_percentile = [i / self.founded_papers_count[-1] * 100 for i in self.founded_papers_count]
        self.total_papers_count.insert(0, 0)
        founded_papers_percentile.insert(0, 0)
        return [self.total_papers_count, founded_papers_percentile]
