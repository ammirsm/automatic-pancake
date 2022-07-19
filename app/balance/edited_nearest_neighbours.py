from imblearn.under_sampling import EditedNearestNeighbours as ENN

from .base import SamplerBase


class EditedNearestNeighbours(SamplerBase):
    def __init__(self):
        sampler = ENN(random_state=42)
        super().__init__(sampler=sampler)
