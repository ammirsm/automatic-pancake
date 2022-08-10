# Auto Pancake Agent

[comment]: <> (Paper description)

## Overview
This repository serves as a test platform for the active learning agent for the systematic review of scientific publications. We will require a RIS file with the location to the PDFs and meta-data of your papers that you wish to add to your database for this.

You will require the votes for title screening and the votes for the full-text screening decision for simulation purposes, which you must provide into the system as a dataset.

Classifiers, feature extractions, decisions, and query strategies will all be handled by separate modules in the system.

The simulations will generate a file with the system's results. The findings have been shown on the [Tabluea Dashboard](https://hubmeta.com/exploring-ai).

If you wish to use this package in production, you will just need the classes and functions, which you may incorporate into your application.

## Setup

### Docker Setup
You can install docker and build up the repository using docker-compose.
``` shell
$ docker compose up
```
### Regular Setup
Another way for setting up the repository will be using traditional virtual environment setup as below.

``` shell
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Usage


### Feed Data
[comment]: <> (### You need to prepare data which will be pass to the other file)
For configuration of your active learning agent you can use the following parts.

### Prepare config
To run the simulation, you'll need a sample configuration and dataset. You may look at the example structure and design your own to put up your own setup and dataset.

``` shell
$ cp sample_files/configs/sample_configs.json ./app/configs.json
```
### Run simulation
In your docker container or Python environment, use the following command to run simulation based on created dataset and config.
``` shell
$ python main.py
```

### Run parallel simulation
Running several simulations, such as those in the article, may require a lot of server time, specifically if they are done sequentially. We created a tool for parallelizing simulations for this purpose. To do this, place the configurations you wish to execute in parallel under `/parallel run/json configs/` and it will run each configuration as a distinct process.

For parallel setup, you must first establish a config directory and a default.env file.
``` shell
$ mkdir -p ./parallel_run/json_configs/
$ cp parallel_run/parallel_config.env.sample parallel_run/parallel_config.env
```
Then, transfer the example configs file to the newly created directory.
``` shell
$ cp sample_files/configs/parallel_configs/* ./parallel_run/json_configs/
```
Then you can run them parallelly

``` shell
$ ./parallel_run/parallel_run.sh
```
### Export result
Your results will be stored in the directory `./results/`.

There is an option to use the results and convert them to a format suitable for tabluea and other visualisation tools, such as an excel file. Look at `./app/result processing utils` to see how to achieve this.

Some visualisation features are available for the findings. The `draw.py` file contains all of the routines.

## Prepare your dataset
use our data_processing scripts to convert your ris file and pdfs into the csv file.

then you should add two columns (title_label, fulltext_label) to generated dataset.

## Development
For pushing something on our codebase you need to clone our repository:
``` shell
$ git clone github.com:ammirsm/auto-pancake-agent
```

We're using [pre-commit](https://pre-commit.com/) to lint our code.
``` shell
$ pre-commit install
```
You must perform pre-commit linting before pushing any changes or issuing merge requests.

## Further development
[ ] ### You can add more features to the agent

## Citation
### link to the paper and the visualization
