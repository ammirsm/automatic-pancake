class SamplerBase:
    def __init__(self, sampler):
        self.sampler = sampler

    def fit_resample(self, training_set, label_set):
        return self.sampler.fit_resample(training_set, label_set)
