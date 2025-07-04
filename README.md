# MDFC

## Run a Training

1. You need to be in a Python 3.10 environment

2. Install dependencies:  
   `pip install -r MPCP/requirements.txt`

3. Set up a config file (see section "Config File Setup")

4. Run the training:  
   `python3.10 -m MDFC.RUN.run_training config/file/path.json`  
   - It's also recommended to use `CUDA_VISIBLE_DEVICES=<device_number>` and `nohup` options

5. Your model, logs, and results will be saved in the output folder

## Config File Setup

The config file is a JSON file.

Example config file:

```json
{
    "name": "test2",
    "train_file": "~/train2/*",
    "valid_file": "~/test2/*",
    "num_interactions": 2,
    "num_elements": 1,
    "max_num_epochs": 20000,
    "plot_frequency": 100,
    "batch_size": 3,
    "default_dtype": "float32",
    "cutt_off": 2.0,
    "num_channels": 32,
    "restart_latest": true,
    "F_mul": 1000
}

To view config descriptions, run: "python3.10 MDFC.py --config_list_training" or "'python3.10 MDFC.py --config' + *name of the option*"

# Code Overview

## Functions

Contains all the building blocks needed by the model to run, from embedding to optimization.

## model

Contains all the components used to build and run the model.

## RUN

Contains the main functions you should use to execute the code.

## tools

Contains utility functions for non-AI tasks like plotting, etc.

# Python package version of the code

The package version of the code is functional but hasn't been thoroughly tested. It may require some adjustments. Until proper testing is completed, please use the standard version of the code.

The pyproject.toml file is used to compile the code into a package.