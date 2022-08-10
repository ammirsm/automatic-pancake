import ssl

import gensim
import nltk
import numpy as np
import pandas as pd
import spacy
from langdetect import detect
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from app.import_export import export_data, import_data


class PreprocessBase:
    def __init__(self, data_path, name):
        self.name = name
        self.nlp_spacy = None
        self.stemmer = None
        self.stop_words = None
        self.data = import_data(data_path)
        # self.data = self.data[:200]
        self.data = pd.DataFrame(self.data)

    def init_spacy(self):
        """
        Initialize spacy model
        """
        self.nlp_spacy = spacy.load("en_core_web_sm")

    def init_stemmer(self):
        self.stemmer = SnowballStemmer("english")

    def init_stop_words(self):
        from app.data_processing.stopwords import ScientificStopWords

        self.stop_words = ScientificStopWords()
        self.stop_words = self.stop_words.get_stop_words()

    def nltk_init(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        np.random.seed(2018)

        nltk.download("wordnet")
        nltk.download("omw-1.4")


class PreprocessTFIDF(PreprocessBase):
    max_token_length = 3
    report_obj = {}

    def __init__(self, data_path, name):
        super().__init__(data_path, name)
        self.nltk_init()
        self.init_spacy()
        self.init_stop_words()
        self.init_stemmer()

    def process(self):
        self.data.fillna("", inplace=True)

        # merge titles
        (
            self.report_obj["ris_filled_title"],
            self.report_obj["crossref_filled_title"],
        ) = self._merge_two_columns("ris-title", "crossref-name", "title")
        self._lower_and_remove_punctuation("title")

        # merge abstracts
        (
            self.report_obj["ris_filled_abstract"],
            self.report_obj["crossref_filled_abstract"],
        ) = self._merge_two_columns("ris-abstract", "crossref-abstract", "abstract")
        self._lower_and_remove_punctuation("abstract")

        # create processed columns
        self._preprocess_data_columns(
            {
                "title": "processed_title",
                "abstract": "processed_abstract",
                "endnote-pdf_text": "processed_endnote_pdf_text",
                "pdf_manual-pdf_text": "processed_pdf_manual_pdf_text",
            }
        )

        self.data["processed_metadata"] = (
            self.data["processed_title"] + " " + self.data["processed_abstract"]
        )

        self.data["meta_data"] = self.data["title"] + " " + self.data["abstract"]

        # create language column based on title and abstract
        self._language_detection("meta_data", "language")

    def export(self, output_path):
        export_data(self.data, f"{output_path}/{self.name}.pickle")
        self.data.to_csv(f"{output_path}/{self.name}.csv")

    def report(self):
        """
        update report the dataframe
        """
        report = {
            "total_papers": len(self.data),
            "total_paper_english": len(self.data[self.data["language"] == "en"]),
            "duplicate_papers": len(self.data[self.data["is_duplicate"] == 1]),
            "fulltext_accepted": len(self.data[self.data["fulltext_label"] == 1]),
            "title_accepted": len(self.data[self.data["title_label"] == 1]),
            "libkey_founded": len(
                self.data[
                    (self.data["libkey-fullTextFile"] != "")
                    | (self.data["libkey-openAccess"] != "")
                ]
            ),
            "libkey_open_access": len(
                self.data[
                    (self.data["libkey-openAccess"] != "")
                    & (self.data["libkey-openAccess"])
                ]
            ),
            "libkey_fulltext_available": len(
                self.data[self.data["libkey-fullTextFile"] != ""]
            ),
            "crossref_founded": len(self.data[self.data["crossref-response"] != ""]),
            "endnote_papers_founded": len(
                self.data[self.data["endnote-pdf_text"] != ""]
            ),
            "pdf_manual_founded": len(
                self.data[self.data["pdf_manual-pdf_text"] != ""]
            ),
        }
        self.report_obj = {**self.report_obj, **report}

    def _merge_two_columns(self, column_1, column_2, target_column, empty_value=""):
        if target_column not in self.data.columns:
            self.data[target_column] = empty_value

        filled_with_column_1 = 0
        filled_with_column_2 = 0
        for index, row in self.data.iterrows():
            if column_1 in self.data.columns and row[column_1] != empty_value:
                self.data.loc[index, target_column] = row[column_1]
                filled_with_column_1 += 1
            elif column_2 in self.data.columns and row[column_2] != empty_value:
                self.data.loc[index, target_column] = row[column_2]
                filled_with_column_2 += 1
        return filled_with_column_1, filled_with_column_2

    def _lower_and_remove_punctuation(self, target_column):
        # lower
        self.data[target_column] = self.data[target_column].str.lower()
        # remove spaces
        self.data[target_column] = self.data[target_column].str.strip()
        # remove punctuation
        self.data[target_column] = self.data[target_column].str.replace(r"[^\w\s]", "")
        # remove tags
        self.data[target_column] = self.data[target_column].str.replace(
            "<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});", ""
        )

    def _stemming(self, text):
        """
        lemmatize and stemming the text for the dataframe and return the lemmatized text
        It will use the WordNetLemmatizer and SnowballStemmer from nltk
        :param text: the text to be processed
        :return: the processed text

        """
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos="v"))

    def _preprocess(self, text):
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
                token not in self.stop_words
                and len(token) > self.get_max_token_length()
            ):
                result.append(self._stemming(token))
        return " ".join(result)

    def _preprocess_data_columns(self, column_names):
        """
        preprocess data columns in the dataframe
        :param column_names: dictionary of the column names and their target column names
        :return: the dataframe with the preprocessed columns\
        sample column_names = {'title': 'title_preprocessed', 'description': 'description_preprocessed'}
        """
        for read, write in column_names.items():
            if read not in self.data.columns:
                continue
            self.data[write] = self.data[read]
            self.data[write] = self.data[write].apply(self._convert_to_unicode)
            self.data[write] = self.data[write].apply(self._preprocess_semantic_text)
            self._lower_and_remove_punctuation(write)
            self.data[write] = self.data[write].apply(self._preprocess)
            self.data[write] = self.data[write].apply(self._preprocess)

        return self.data

    def _convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def _preprocess_semantic_text(self, text):
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

    def _langdetect_helper(self, text):
        """
        detect the language of the text
        """
        if not text or text.strip() == "":
            return ""
        return detect(text)

    def _language_detection(self, target_column, language_column_name):
        """
        detect the language of the text in the target column and add the language to the language column
        :param target_column: the target column to detect the language
        :param language_column_name: the name of the language column
        """
        self.data[language_column_name] = self.data[target_column].apply(
            self._langdetect_helper
        )

    def set_max_token_length(self, token_length):
        self.max_token_length = token_length

    def get_max_token_length(self):
        return self.max_token_length
