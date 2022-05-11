import gensim

from app.configs import stop_words_path


class ScientificStopWords:
    """
    A class to hold the stop words for the scientific domain.
    source is
    https://github.com/seinecle/Stopwords/tree/master/src/main/java/net/clementlevallois/stopwords/resources
    """

    def __init__(self):
        for file in stop_words_path:
            with open(file, "r") as f:
                stop_words = f.read().splitlines()

        self.stop_words = gensim.parsing.preprocessing.STOPWORDS.union(set(stop_words))

    def get_stop_words(self):
        return self.stop_words
