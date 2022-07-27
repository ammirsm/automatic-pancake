from .adasyn import ADASYN
from .condensed_nearest_neighbour import CondensedNearestNeighbour
from .edited_nearest_neighbours import EditedNearestNeighbours
from .near_miss import NearMiss
from .random_over import RandomOverSampler
from .random_under import RandomUnderSampler
from .smote import SMOTE


def get_sampler_class(sampler):
    mapper = {
        "CondensedNearestNeighbour": CondensedNearestNeighbour,
        "EditedNearestNeighbours": EditedNearestNeighbours,
        "ADASYN": ADASYN,
        "NearMiss": NearMiss,
        "RandomOverSampler": RandomOverSampler,
        "RandomUnderSampler": RandomUnderSampler,
        "SMOTE": SMOTE,
    }
    return mapper.get(sampler)
