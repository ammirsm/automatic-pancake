from .max_prob import MaxProb
from .max_uncertainty import Uncertainty
from .mix_checking import MixChecking

# from .auto_mixed_sd import AutoMixedSD
# from .max_prob_with_prioritize_fulltext import MaxProbPrioritizeFulltext
# from .mix_checking_inside import MixCheckingInside
# from .title import Title


def get_query_class(name):
    mapper = {
        "max_prob": MaxProb,
        "uncertainty": Uncertainty,
        "mix_checking": MixChecking,
    }
    return mapper.get(name)
