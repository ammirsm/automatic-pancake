from imblearn.over_sampling import RandomOverSampler as RandomOverSampler_imblearn

from .base import SamplerBase


class RandomOverSampler(SamplerBase):
    def __init__(self):
        sampler = RandomOverSampler_imblearn(random_state=42)
        super().__init__(sampler=sampler)
