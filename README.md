# Auto Pancake Agent Simulation :pancakes: :robot: :ghost:
## Systematic review active learning agent simulation
![automatic pancake](https://github.com/ammirsm/automatic-pancake/blob/main/automatic-pancake.jpg?raw=true)
*Note: This repository is not about [Chefstack Automatic Pancake Machine](https://uncrate.com/chefstack-automatic-pancake-machine/) and we're just using it as a sample of automatic pancake machine :laughing:.*



## Overview
This repository serves as a test platform for the active learning agent for the systematic review of scientific publications. We will require a RIS file with the location to the PDFs and meta-data of your papers that you wish to add to your database for this.

You will require the votes for title screening and the votes for the full-text screening decision for simulation purposes, which you must provide into the system as a dataset.

Classifiers, feature extractions, decisions, and query strategies will all be handled by separate modules in the system.

The simulations will generate a file with the system's results. The findings have been shown on the [Tabluea Dashboard](https://hubmeta.com/exploring-ai).

If you wish to use this package in production, you will just need the classes and functions, which you may incorporate into your application.

![simulation process](https://github.com/ammirsm/automatic-pancake/blob/main/simulation.jpg?raw=true)
*Note: This picture has came from the paper, for more information about the detail of the process take a look at the paper.*


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
Running several simulations, such as those in the article, may require a lot of server time, specifically if they are done sequentially. We created a tool for parallelizing simulations for this purpose. To do this, place the configurations you wish to execute in parallel under `/parallel_run/json_configs/` and it will run each configuration as a distinct process.

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

There is an option to use the results and convert them to a format suitable for tabluea and other visualisation tools, such as an excel file. Look at `./app/result_processing_utils` to see how to achieve this.

Some visualisation features are available for the findings. The `draw.py` file contains all of the routines.

## Prepare your dataset
To prepare your dataset for active learning, use `data_processing.py` to clean the data and make it ready.
This data processing.py script can be found in the repository and will convert your RIS and PDF files to a CSV file suited for active learning.
When you generate the dataset for the active learning agent, you will need to add two extra columns that reflect your screening votes. For the results of the title and abstract screening (stage 1), enter your decisions as 1 and 0 (accept or reject) in the column name `title_label`, and for the fulltext screening, enter your decisions as 1 and 0 (accept or reject) in the column name `fulltext_label`.

For more information about the data processing, please see the README.md file in the data_processing directory. [README](https://github.com/ammirsm/automatic-pancake/blob/main/app/data_processing/README.md)
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
### Visualization
You can take a look at our visualization at ([HubMeta](https://hubmeta.com/exploring-ai)) website.

### Cite our paper
Cite us with:
```
@article{Saeidmehr2024SystematicRU,
  title={Systematic review using a spiral approach with machine learning},
  author={Amirhossein Saeidmehr and Piers Steel and Faramarz Famil Samavati},
  journal={Systematic Reviews},
  year={2024},
  volume={13},
  url={https://api.semanticscholar.org/CorpusID:267019792}
}
```


