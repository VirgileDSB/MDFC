###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/mace/cli/run_train.py
#       First Authors: Ilyes Batatia, Gregor Simm, David Kovacs
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

import json
import logging
import sys
import os

from ..Functions.model_optimisation.Loss import projected_mean_squared_error, euclidian_squared_error
from ..Functions.model_optimisation.get_optimizer import get_optimizer
from ..Functions import interaction

from ..model.MPCP_model import MPCP
from ..model.train_model import train

from ..tools.torch_geometric.data import Data
from ..tools.checkpoint import CheckpointHandler, CheckpointState
from ..tools.LR_Scheduler import LRScheduler
from ..tools.visualise_train import TrainingPlotter, print_comparaison_file
from ..tools import torch_geometric
from ..tools.error_table import create_error_table
from ..tools.config_tools import update_config_training, config_class_training, config_load_model
from ..tools.logger import setup_logger, MetricsLogger
from ..tools.dataset import CustomDataset

from torch_ema import ExponentialMovingAverage
import torch

import numpy as np
from typing import Optional
from copy import deepcopy
from dacite import from_dict
from pathlib import Path


def run(config_file_path : str, name: str) -> None:
    """
    This script runs the training for model/MPCP_model.py using the configuration file provided in config_file_path.
    """

    ################### Config Loading ######################

    config: config_class_training

    # Open and read config file
    with open(config_file_path, 'r') as file:
        Config = json.load(file)

    # Update config with default
    Config, model_config, unknow_keys = update_config_training(Config)
    config = from_dict(data_class=config_class_training, data=Config)
    model_config = from_dict(data_class=config_load_model, data=model_config)

    tag = f"{config.name}_run-{config.seed}" #Tag of the training

    # Setup seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Setup logger
    setup_logger(level_print_file=config.file_log_level, level_print_terminal=config.terminal_log_level, tag=tag, directory=config.log_dir)
    logger = logging.getLogger("Main_Logger")

    # Print config messages in logger
    logger.info("===========VERIFYING SETTINGS===========")

    if unknow_keys:
        logger.warning(f"Error : unknow config key : {', '.join(unknow_keys)}")
    
    # Set type: "float64" for better presision, "float32" for fast models
    torch.set_default_dtype(getattr(torch, config.default_dtype))

    # Init_device (only use cuda for serious training)
    if config.device != "cuda":
        logger.CRITICAL(f"only CUDA device has been tested, cuequivariance_torch may not work with other devices")
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
    train_dataset = CustomDataset(config.train_file, config.cutt_off, config.num_elements, config.default_dtype, config.F_mul)

    if train_dataset.__len__() == 0:
        raise ValueError(f"No training datasets found")
    logger.debug(f"Successfully loaded training dataset with {train_dataset.__len__()} samples")

    valid_dataset = CustomDataset(config.valid_file, config.cutt_off, config.num_elements, config.default_dtype, config.F_mul)

    if train_dataset.__len__() == 0:
        raise ValueError(f"No valid datasets found")
    logger.debug(f"Successfully loaded validation dataset with {valid_dataset.__len__()} samples")

    dataset_size = train_dataset.__len__()

    logger.info(f"training dataset size: {dataset_size}")



    # Data loading managed by torch_geometric
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        sampler=None,
        shuffle=True,
        drop_last=True,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )

    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=valid_dataset,
        batch_size=config.valid_batch_size,
        sampler=None,
        shuffle=False,
        drop_last=False,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )

    # For testing purpose only
    if config.test_infos:
        first_batch = next(iter(train_loader))
        x_0 = first_batch.x[(first_batch.batch == 0)]
        edge_mask = (first_batch.edge_index[0] < x_0.shape[0]) & (first_batch.edge_index[1] < x_0.shape[0])
        edge_index_0 = first_batch.edge_index[:, edge_mask]
        data0 = Data(x=x_0, edge_index=edge_index_0)

        logger.info(f"number of batchs : {len(train_loader)}")
        logger.info(f"one batch shape: {first_batch}")
        logger.info(f"nodes features (x) of the 1st data in 1st batch: {data0.x}")
        logger.info(f"edge_index of the 1st data in 1st batch: {data0.edge_index}")


    ################### Model ######################


    # Loss function
    # Any type of loss functions can be implemented here, fell free to do so
    # This statement governs the loss behavior throughout the code, changing it updates the entire loss function.
    # A loss function juste need to be a class with the same 'shape' as euclidian_squared_error
    if config.loss_fct == "projected_mean_squared_error":
        loss_fn = projected_mean_squared_error()
    elif config.loss_fct == "euclidian_mean_squared_error":
        loss_fn = euclidian_squared_error(train_loader)
    else:
        logger.CRITICAL(f"Unknow Loss function name {config.loss_fn}, applying euclidian_squared_error_force")
        loss_fn = projected_mean_squared_error()


    # Logger infos
    logger.info("===========MODEL DETAILS===========")
    logger.info("Building model")
    logger.info(
        f"Message passing with {config.num_channels} channels and max_L={config.max_L}"
    )
    logger.info(f"Hidden irreps: {config.hidden_irreps}")
    logger.info(
        f"{config.num_interactions} layers, each with correlation order: {config.correlation} and spherical harmonics up to: l={config.max_ell}"
    )
    logger.info(
        f"{config.num_radial_basis} radial and {config.num_polynomial_cutoff} basis functions"
    )
    logger.info(
        f"Radial cutoff: {config.cutt_off}"
    )
    logger.info(f"Hidden irreps: {config.hidden_irreps}")

    # Model setup
    # interaction_cls_first is the interaction type (see documentation) for the 1st message passing procass
    # interaction_cls is for all the others
    model = MPCP(
        config = model_config,
        interaction_cls_first = interaction.interaction_classes[config.interaction_first],
        interaction_cls = interaction.interaction_classes[config.interaction],
    )

    print(f"\033[34m Model built with {int(sum(np.prod(p.shape) for p in model.parameters()))} parameters \033[0m")

    model.to(config.device)

    # Print the model into logger file, can be commented for more compact logger file
    logger.debug(model)
    logger.info(f"Total number of parameters: {int(sum(np.prod(p.shape) for p in model.parameters()))}")
    logger.info("")
    logger.info("===========OPTIMIZER INFORMATION===========")
    logger.info(f"Using {config.optimizer.upper()} as parameter optimizer")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Number of gradient updates: {int(config.max_num_epochs*len(train_dataset)/config.batch_size)}")
    logger.info(f"Learning rate: {config.lr}, weight decay: {config.weight_decay}")
    logger.info(loss_fn)

    # Optimizer
    optimizer: torch.optim.Optimizer
    optimizer = get_optimizer(config, model)


    ################### Model options ######################

    # Result file logger
    results_logger = MetricsLogger(directory=config.results_dir, tag=tag + "_train")

    # Class that is called during training to dynamically adjust the learning rate and improve convergence and performance.
    # Do not modify without understanding
    lr_scheduler = LRScheduler(optimizer, config)


    # Class to save and load '.model' files
    checkpoint_handler = CheckpointHandler(
        directory=config.checkpoints_dir,
        tag=tag,
        keep=config.keep_checkpoints,
    )

    # Restart a previous training
    start_epoch = 0
    if config.restart_latest:
        opt_start_epoch = checkpoint_handler.load_latest(
            state = CheckpointState(model, optimizer, lr_scheduler),
            device = config.device,
        )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    # Not tested yet !!
    # Exponential Moving Average (EMA) setup. 
    # EMA Stabilizing model convergence by adding a weight in old values to bring some 'memory'
    # 'decay' close to 1.0 means slow updates and more smoothing (recomended)
    # See torch EMAHandler docs
    ema: Optional[ExponentialMovingAverage] = None
    if config.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_decay)
    else:
        for group in optimizer.param_groups:
            group["lr"] = config.lr


    ################### Training Plotter ######################

    # For plotting purposes
    train_valid_data_loader = {"train": train_loader, "valid": valid_loader}

    # Setup the plotter for train() function. If train_frequency is set to 0, no training plots
    if config.plot and config.plot_frequency > 0:
        try:
            plotter = TrainingPlotter(
                plot_dir = config.plot_dir,
                tag = tag + "_training",
                config = config,
                metrics_to_print = config.metrics_to_print,
                train_valid_data = train_valid_data_loader,
                test_data = {},
                device = config.device,
                plot_frequency = config.plot_frequency,
                )
        except Exception as e:
            logger.warning(f"Creating Plotter failed: {e}")
            plotter = None
    else:
        plotter = None

    ################### Training ######################

    print()
    print()
    print("===========STARTING TRAINING===========")
    print()
    print()

    last_epoch = train(
        model=model,
        config = config,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        start_epoch=start_epoch,
        results_logger = results_logger,
        device=config.device,
        ema=ema,
        plotter=plotter,
    )


    logger.info("")
    logger.info("===========RESULTS===========")

    #################### Testing ######################

    # This has not been tested yet !!
    # May need some modifications to work properly but should be fine 

    model_to_evaluate = model
    test_sets = {}
    test_data_loaders = {}
    suffix = "" # A suffix option that will be implemented later
    if config.test_dir is not None:
        # Get files with suffix
        test_files = [os.path.join(os.path.expanduser(config.test_dir), f) for f in os.listdir(os.path.expanduser(config.test_dir)) if f.endswith(suffix)]
        for test_file in test_files:
            name = os.path.splitext(os.path.basename(test_file))[0] # Extract name of the file from path of the file
            if not "result" in name:
                test_sets[name] = CustomDataset(test_file, config.cutt_off, config.num_elements, config.default_dtype, config.F_mul) # Store tests in a test dict


        try:
            drop_last = test_set.drop_last
        except AttributeError as e: 
            drop_last = False


        one_test_loader = torch_geometric.dataloader.DataLoader(
            test_set,
            batch_size=config.valid_batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        test_data_loaders[test_name] = one_test_loader

        tables_test = {}
        for name, test_data_loader in test_data_loaders.items():
            tables_test[name] = create_error_table(
                metrics_to_print=config.metrics_to_print,
                data_loader=test_data_loader,
                model=model_to_evaluate,
                loss_fn=loss_fn,
                device=config.device,
            )
            logger.info(f"Error-table on TEST {name}:\n" + str(tables_test[name]))

        print_comparaison_file(    
            model = model,
            tables_test = tables_test,
            data_loader = test_data_loaders,
            test_dir = config.test_dir,
            device = config.device,
        )

    ############## Save model and metrics ##############

    # todo: implement swa


    # Save entire model
    model_path = Path(config.checkpoints_dir) / (tag + ".model")
    logger.info(f"Saving model to {model_path}")
    model_to_save = deepcopy(model) # Independent copy of the model
    torch.save(model_to_save, model_path)

    # Save compiled model (not tested yet)
    try:
        path_complied = Path(config.model_dir) / (config.name + "_compiled.model")
        logger.info(f"Compiling model, saving metadata to {path_complied}")
        model_compiled = torch.compile(deepcopy(model_to_save))
        torch.jit.save(model_compiled, path_complied)

    except Exception as e:
        logger.debug(f"compiled model failed: {e}")
        pass

    logger.info("Computing metrics for training, validation, and test sets")
    for param in model.parameters():
        param.requires_grad = False


    ##################### Final plotter ######################

    if config.plot:
        try:
            plotter = TrainingPlotter(
                plot_dir = config.plot_dir,
                config = config,
                tag = tag + "_final",
                metrics_to_print = config.metrics_to_print,
                train_valid_data = train_valid_data_loader,
                test_data = valid_loader,
                device = config.device,
                plot_frequency = config.plot_frequency,
            )
            plotter.plot(last_epoch, model_to_evaluate)
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

    ##################### Finishing and quit ######################

    logger.info("Done")








if __name__ == "__main__": #run main whene run_training is called
    if len(sys.argv) == 2:
        config_file_path = sys.argv[1]
        name = None
    elif len(sys.argv) == 3:
        config_file_path = sys.argv[1]
        name = sys.argv[2]
    else:
        print("Error:\n Command line arguments: <config_file_path> Optional[<name>]")
        sys.exit(1)

    torch.cuda.empty_cache()
    run(config_file_path, name)

