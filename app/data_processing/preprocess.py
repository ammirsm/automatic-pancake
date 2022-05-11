import ssl

import gensim
import nltk
import numpy as np
import pandas as pd
import spacy
from langdetect import detect
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from app.import_export import import_data


def nltk_init():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    np.random.seed(2018)

    nltk.download("wordnet")
    nltk.download("omw-1.4")


class PreprocessBase:
    def __init__(self, data_path):
        self.nlp_spacy = None
        self.stemmer = None
        self.data = import_data(data_path)
        self.data = pd.DataFrame(self.data)

    def init_spacy(self):
        """
        Initialize spacy model
        """
        self.nlp_spacy = spacy.load("en_core_web_sm")

    def init_stemmer(self):
        self.stemmer = SnowballStemmer("english")


class PreprocessTFIDF(PreprocessBase):
    max_token_length = 3

    def __init__(self, data_path):
        super().__init__(data_path)
        nltk_init()
        self.init_spacy()

    def process(self):
        self.data = self.data.fillna("", inplace=True)

        pass

    def stemming(self, text):
        """
        lemmatize and stemming the text for the dataframe and return the lemmatized text
        It will use the WordNetLemmatizer and SnowballStemmer from nltk
        :param text: the text to be processed
        :return: the processed text

        """
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos="v"))

    def preprocess(self, text):
        """
        preprocess the text for the dataframe
        check if the text is stop word or not and if the token length is greater than the max token length
        :param text: the text to be processed in the dataframe
        :return: the processed text
        """
        text = str(text)
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if (
                token not in gensim.parsing.preprocessing.STOPWORDS
                and len(token) > self.get_max_token_length()
            ):
                result.append(self.stemming(token))
        return " ".join(result)

    def preprocess_data_columns(self, column_names):
        """
        preprocess data columns in the dataframe
        :param column_names: dictionary of the column names and their target column names
        :return: the dataframe with the preprocessed columns\
        sample column_names = {'title': 'title_preprocessed', 'description': 'description_preprocessed'}
        """
        for read, write in column_names.items():
            self.data[write] = self.data[read].apply(self.preprocess)
        return self.data

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def preprocess_semantic_text(self, text):
        """
        preprocess the text for the dataframe with spacy and remove unrelated tokens
        This will use the spacy library to preprocess the text and remove emails and urls and digits and currencies
        :param text: the text to be processed
        :return: the processed text
        """
        self.nlp_spacy.max_length = 10000000
        doc = self.nlp_spacy(text)
        doc_new = []
        for i, token in enumerate(doc):
            if (
                token.like_email
                or token.like_num
                or token.like_url
                or token.is_digit
                or token.is_title
                or token.is_currency
                or token.ent_type_ in ["ORG", "MONEY", "GPE"]
            ):
                pass
            else:
                doc_new.append(token.norm_)
        return " ".join(doc_new)

    def langdetect_helper(self, text):
        """
        detect the language of the text
        """
        return detect(text)

    def language_detection(self, target_column, language_column_name):
        """
        detect the language of the text in the target column and add the language to the language column
        :param target_column: the target column to detect the language
        :param language_column_name: the name of the language column
        """
        self.data[language_column_name] = self.data[target_column].apply(
            self.langdetect
        )

    def set_max_token_length(self, token_length):
        self.max_token_length = token_length

    def get_max_token_length(self):
        return self.max_token_length
