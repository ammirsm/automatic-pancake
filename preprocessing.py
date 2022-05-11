import ssl

import gensim
import nltk
import numpy as np
import pandas as pd
import syntok.segmenter as segmenter
from crossref.restful import Works
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

np.random.seed(2018)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("wordnet")
nltk.download("omw-1.4")

model = SentenceTransformer("https://huggingface.co/gsarti/scibert-nli")


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


def check_the_file_is_available_or_not(file_name):
    try:
        f = open(file_name, "r")
        f.close()
        return True
    except FileNotFoundError:
        return False


def pdftotext_reader(file_name):
    print(file_name)
    import pdftotext

    if check_the_file_is_available_or_not(file_name):
        with open(file_name, "rb") as f:
            pdf = pdftotext.PDF(f)
        return " ".join(pdf)
    return ""


def text_to_vector(document):
    global model
    if not document:
        return None
    paragraphs = []
    for paragraph in segmenter.process(document):
        sentences = []
        for sentence in paragraph:
            sentences.append(" ".join([token.value for token in sentence]).lower())
        embeddings = model.encode(sentences)
        embeddings = embeddings.mean(0)
        paragraphs.append(embeddings)
    paragraphs = np.array(paragraphs)
    paragraphs = paragraphs.mean(0)
    return paragraphs


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def update_pdfs_text(ris_file_name, pdf_path_base, my_data):
    my_data["keywords_new_whole"] = np.nan
    import rispy

    ris_file = open(ris_file_name, "r")
    ris_file = rispy.load(ris_file)
    ris_file = [entry for entry in ris_file]
    ris_data = pd.DataFrame(ris_file)
    for i, data in enumerate(ris_data["title"]):
        # my_data[my_data["title"] == data]["file"].values[0/]
        path = ris_data.at[i, "file_attachments1"].split("//")[1]
        whole_path = pdf_path_base + "/" + path
        my_data.loc[my_data["title"] == data, "keywords_new_whole"] = pdftotext_reader(
            whole_path
        )


def update_pdfs_text_new(my_data):
    for i, row in my_data.iterrows():
        data = []
        for row_name in ["title", "abstract", "keywords_new_whole"]:
            if not row[row_name] == "":
                data.append(text_to_vector(row[row_name]))
        data = np.array(data)
        data = data.mean(0)
        my_data.loc[i, "keywords_new_whole_bert"] = data


def preprocess_pdf_text(text):
    text = text.split("\n")
    text = [i.strip() for i in text]
    text_new = []
    for i in text:
        i_new = i.split()
        if len(i_new) > 4:
            text_new.append(i)
    text = " ".join(text_new)
    import re

    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    return text


def preprocess_semantic_text(text):
    import spacy

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 10000000
    doc = nlp(text)
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


def preprocess_data(df, column_name):
    df = df.fillna("", inplace=True)
    df[column_name] = df[column_name].apply(preprocess_pdf_text)
    df[column_name] = df[column_name].apply(preprocess_semantic_text)
    # df[column_name] = df[column_name].apply(preprocess)


def langdetect(text):
    from langdetect import detect

    return detect(text)


def langdetect_data(df, column_name):
    df["lang"] = df[column_name].apply(langdetect)


works = Works()


def get_abstract_from_crossref(doi):
    print(doi)
    if not doi:
        return ""
    global works
    data = works.doi(doi)
    if not data:
        return ""
    if "abstract" in data:
        return data["abstract"]
    else:
        return ""
