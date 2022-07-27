from imblearn.over_sampling import SMOTE as SMOTE_imblearn

from .base import SamplerBase


class SMOTE(SamplerBase):
    def __init__(self):
        sampler = SMOTE_imblearn(random_state=42)
        super().__init__(sampler=sampler)
