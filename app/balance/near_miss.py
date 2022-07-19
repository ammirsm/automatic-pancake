from imblearn.under_sampling import NearMiss as NearMiss_imblearn

from .base import SamplerBase


class NearMiss(SamplerBase):
    def __init__(self):
        sampler = NearMiss_imblearn(random_state=42)
        super().__init__(sampler=sampler)
