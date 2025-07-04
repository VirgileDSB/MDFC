###########################################################################################
#
# First Authors: Virgile de Saint Blanquat
#
# This script loads a model and runs it on a single data file.
# 99% of the script running time is spent for model loading.
# An major improvement for this script could thene be to:
#   -Load the model only once
#   -'pause' the script until the user (or a external python script) specify a file to run
#   -Run the script on the specified file
#   -'pause' the script again until the user specify a new file to run or decide to exit
#
# A multiple datas files loading option should also be added. It should be pretty easy to do with a loop on files in a folder/list.
#
###########################################################################################

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from dacite import from_dict

from ..Functions import interaction
from ..model.MPCP_model import MPCP
from ..tools import torch_geometric
from ..tools.config_tools import config_load_model, update_config_use_model, config_run_model
from ..tools.dataset import CustomDataset
from ..tools.logger import  setup_logger
from ..tools.printing import print_output_file

def run(config_file_path : str, file_path : str) -> None:
    """
    This script runs the model for one data without training 
    """

    ################### Config Loading ######################

    config_model: config_load_model # This class is a dataclass that contains the configurations options to load the model

    config: config_run_model # This class is a dataclass that contains the others configurations options (like the model path, the device, etc.)

    # Open and read config file
    try:
        with open(config_file_path, 'r') as file:
            Config = json.load(file) # Config is a dict with only input in the file for now
    except FileNotFoundError:
        print(f"Config file {config_file_path} not found.")
        sys.exit(1)

    Config, config_model, unknow_keys = update_config_use_model(Config) # See tools/config_tools.py for more details
    # Config and config_model are now dicts with all the options uptdated
    config_model = from_dict(data_class=config_load_model, data=config_model) # Transform the dict into a dataclass for more convenient access
    config = from_dict(data_class=config_run_model, data=Config) # Transform the dict into a dataclass for more convenient access

    # This is the name argument that can be used whene running the script
    if file_path is not None:
        config.file_path = file_path
        config.name = Path(file_path).stem
    elif config.file_path is not None:
        config.name = Path(config.file_path).stem 
    else:
        raise ValueError("file name not provided, please provide a file name in the config file or as an argument")

    tag = f"{config.name}_out" #Tag of the training (name of the output file)

    # Setup logger
    setup_logger(level_print_file=config.file_log_level, level_print_terminal="ERROR", tag=tag, directory=config.log_dir)
    logger = logging.getLogger("Main_Logger")

    if unknow_keys:
        logger.warning(f"Error : unknow config key : {', '.join(unknow_keys)}")

    # Print config messages in logger
    logger.info("===========VERIFYING SETTINGS===========")

    # Set type: "float64" for better presision, "float32" for fast models
    torch.set_default_dtype(getattr(torch, config.default_dtype))

    # Init_device (only use cuda)
    if config.device != "cuda":
        logger.CRITICAL("only CUDA device has been tested, cuequivariance_torch may not work with other devices")
        torch.device(config.device)
    else:
        assert torch.cuda.is_available(), "No CUDA device available!"
        torch.device(config.device)
        logger.info(f"CUDA loaded, device: {torch.cuda.get_device_name(0)}")


    ################### Loading datas ######################


    logger.info("===========LOADING INPUT DATA===========")

    # CustomDataset is a custom torch_geometric based dataset with is made for .sph like files
    # The dataset manage data importation from files 
    # Fell free to modify it if you have an other data shape
    dataset = CustomDataset(config.file_path, config_model.cutt_off, config_model.num_elements, config.default_dtype, config.F_mul)

    if dataset.__len__() == 0:
        raise ValueError("file not found or empty")
    logger.debug(f"Successfully loaded file")

    # Data loading managed by torch_geometric
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=None,
        num_workers=1,
    )


    ################### Model ######################

    # Model setup
    # interaction_cls_first is the interaction type (see documentation) for the 1st message passing procass
    # interaction_cls is for all the others
    model = MPCP(
        config = config_model,
        interaction_cls_first = interaction.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls = interaction.interaction_classes["RealAgnosticResidualInteractionBlock"],
    )

    config.model_path = Path(config.model_path).expanduser()
    checkpoint = torch.load(f=config.model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)

    model.eval()

    # Logger infos
    logger.info("===========MODEL DETAILS===========")
    logger.info(f"Message passing with {config_model.num_channels} channels and max_L={config.max_L}")
    logger.info(f"Hidden irreps: {config_model.hidden_irreps}")
    logger.info(f"{config_model.num_interactions} layers, each with correlation order: {config_model.correlation} and spherical harmonics up to: l={config_model.max_ell}")
    logger.info(f"{config_model.num_radial_basis} radial and {config_model.num_polynomial_cutoff} basis functions")
    logger.info(f"Radial cutoff: {config_model.cutt_off}")
    logger.info(f"Hidden irreps: {config_model.hidden_irreps}")


    model.to(config.device)

    # Print the model into logger file, can be commented for more compact logger file with debug level
    logger.debug(model)
    logger.info(f"Total number of parameters: {int(sum(np.prod(p.shape) for p in model.parameters()))}")
    logger.info("")

    for batch in loader:
        batch = batch.to(config.device)
        output = model(batch)

    
    print_output_file(dataset = dataset, output = output, config = config, tag = tag)


    logger.info("Done")


if __name__ == "__main__": #run main whene run_training is called
    if len(sys.argv) == 2:
        config_file_path = sys.argv[1]
        file_path = None
    elif len(sys.argv) == 3:
        config_file_path = sys.argv[1]
        file_path = sys.argv[2]
    else:
        print("Error:\n Command line arguments: <config_file_path> Optional[<file_path>]")
        sys.exit(1)

    torch.cuda.empty_cache()
    run(config_file_path, file_path)

