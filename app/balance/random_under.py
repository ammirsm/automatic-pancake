from imblearn.under_sampling import RandomUnderSampler as RandomUnderSampler_imblearn

from .base import SamplerBase


class RandomUnderSampler(SamplerBase):
    def __init__(self):
        sampler = RandomUnderSampler_imblearn(random_state=42)
        super().__init__(sampler=sampler)
