from .bow import Bow
from .scibert import Scibert
from .tfidf import TFidf


def get_feature_extraction_class(tokenizer):
    mapper = {
        "TF-IDF": TFidf,
        "BOW": Bow,
        "Scibert": Scibert,
    }
    return mapper.get(tokenizer)
