##Preprocessing:
In this function we will use some preprocessing tools in NLP to clean the data and make it ready for active learning process.

##Setup
### Installing python dependencies

Make sure you are in the root directory of the project and run the following command:
```shell
$ pip install -r app/data_processing/requirements.txt
$ python -m spacy download en_core_web_sm
```

Note: If you are having trouble installing Python packages. Check that these operating system packages are installed or install them using the following command:

On Debian, Ubuntu, and friends:
```shell
$ sudo apt-get update
$ sudo apt-get install build-essential libpoppler-cpp-dev pkg-config python-dev
```

On Fedora, Red Hat, and friends:
```shell
sudo yum install gcc-c++ pkgconfig poppler-cpp-devel python-devel redhat-rpm-config
```

##Usage
###How to preprocess data
```
$ python data_processing.py /path/to/your/dataset.ris /path/to/your/pdf/dir/ /path/to/your/output/dir/
```
Note: While we're doing the preprocessing we're saving some assets in the `asset` directory which you can use and not reading the PDFs in the main function later.
