# Active learning agent
TODO:

## Setup
```
$ pip install -r requirements.txt
$ pre-commit install
```

## Download stopwords from the repository

## Preprocess your data with preprocess module
```python
from app.data_processing.preprocess import PreprocessTFIDF
path = "" # your path to the data pickle
path_export = "" # preprocessed data pickle export
data = PreprocessTFIDF(path)
data.process()
data.export(path_export)
data.report()
print(data.report_obj) # report object will contain all of the data
# ```



## TODO LIST
[ ] Preprocessing of the data files cleaning
[ ] Crossref function to get the references
[ ] Cleaning the data and model files


## Van De Dataset
``` python
{'total_papers': 6184, 'total_paper_english': 6152, 'duplicate_papers': 0, 'fulltext_accepted': 43, 'title_accepted': 388, 'libkey_founded': 1318, 'libkey_open_access': 254, 'libkey_fulltext_available': 1156, 'crossref_founded': 1375, 'endnote_papers_founded': 804, 'pdf_manual_founded': 327}
```

## Cultural
```python
{'total_papers': 12107, 'total_paper_english': 10887, 'duplicate_papers': 2768, 'fulltext_accepted': 1379, 'title_accepted': 3080, 'libkey_founded': 3355, 'libkey_open_access': 434, 'libkey_fulltext_available': 2916, 'crossref_founded': 3385, 'endnote_papers_founded': 2165, 'pdf_manual_founded': 6055}
```


## Vandis
```python
{'total_papers': 10953, 'total_paper_english': 10933, 'duplicate_papers': 0, 'fulltext_accepted': 73, 'title_accepted': 806, 'libkey_founded': 6719, 'libkey_open_access': 1530, 'libkey_fulltext_available': 5902, 'crossref_founded': 7030, 'endnote_papers_founded': 1388, 'pdf_manual_founded': 698}
```

## configs

- base lines
-
- new features
