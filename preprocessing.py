import ssl

import gensim
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

np.random.seed(2018)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("wordnet")
nltk.download("omw-1.4")


def lemmatize_stemming(text):
    from nltk.stem.snowball import SnowballStemmer

    englishStemmer = SnowballStemmer("english")
    return englishStemmer.stem(WordNetLemmatizer().lemmatize(text, pos="v"))


def preprocess(text):
    text = str(text)
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return " ".join(result)


def replace_commas_with_space(df, columns):
    for column in columns:
        df[column] = df[column].str.lower().replace(",", " ")


def preprocess_data_column(df, column_names):
    for column_name in column_names:
        df[column_name] = df[column_name].apply(preprocess)


def drop_columns(df, columns):
    df.drop(columns, axis=1, inplace=True)


def write_to_csv(df, file_name):
    df.to_csv(file_name, index=False)


column = ""


def add_column_name_for_df(df, column_names):
    global column
    for column_name in column_names:
        column = column_name
        df[column_name] = df[column_name].apply(add_column_name)


def add_column_name(text):
    try:
        global column
        result = []
        for i in text.split(" "):
            result.append(i)
            result.append(i + "_" + column)
        return " ".join(result)
    except Exception as e:
        print(e)
        return text
