from imblearn.over_sampling import ADASYN as ADASYN_imblearn

from .base import SamplerBase


class ADASYN(SamplerBase):
    def __init__(self):
        sampler = ADASYN_imblearn(random_state=42)
        super().__init__(sampler=sampler)
