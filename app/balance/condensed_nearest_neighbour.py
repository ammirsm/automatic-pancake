from imblearn.under_sampling import CondensedNearestNeighbour as ConNN

from .base import SamplerBase


class CondensedNearestNeighbour(SamplerBase):
    def __init__(self):
        sampler = ConNN(random_state=42)
        super().__init__(sampler=sampler)
