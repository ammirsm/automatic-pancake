# Auto Pancake Agent

[comment]: <> (Paper description)

## Overview

[comment]: <> (discuss how to prepare the data for the agent)

[comment]: <> (discuss steps to build a pancake agent)

[comment]: <> (discuss how to export data)


## Setup

### Docker Setup
run this command and see the result
```
# docker compose up
```
### Regular Setup

```
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ pre-commit install
```

## Usage


### Feed Data
[comment]: <> (### You need to prepare data which will be pass to the other file)

### Prepare config
this is a sample config that use the sample dataset for running simnulation
```
$ cp sample_files/configs/sample_configs.json ./app/configs.json
```
### Run simulation
```
$ python main.py
```

### Run parallel simulation
put your configs in the `./parallel_run/json_configs/`.
it will run each config as a seperated process.

create configs directory and create default .env file for parallel config
```
$ mkdir -p ./parallel_run/json_configs/
$ cp parallel_run/parallel_config.env.sample parallel_run/parallel_config.env
```
copy the sample configs file to the directory
```
$ cp sample_files/configs/parallel_configs/* ./parallel_run/json_configs/
$ ./parallel_run/parallel_run.sh
```

### Export result

## Further development

## Formatting and linting

## Citation
### link to the paper and the visualization
