###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/tools/train.py
#       First Authors: Ilyes Batatia, Gregor Simm, David Kovacs
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

import traceback
import logging
logger = logging.getLogger("Main_Logger")

from contextlib import nullcontext
from typing import Optional
from tqdm import tqdm

import numpy as np
from torch.optim import LBFGS
import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from ..tools.config_tools import config_class_training
from ..tools.visualise_train import TrainingPlotter, Global_class_results

from ..tools.checkpoint import CheckpointHandler, CheckpointState
from ..tools.logger import MetricsLogger
from .evaluate import evaluate
from .take_steps import take_step, take_step_lbfgs

def train(
    model: torch.nn.Module,
    config: config_class_training,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    checkpoint_handler: CheckpointHandler,
    start_epoch: int,
    results_logger: MetricsLogger,
    device: torch.device,
    ema: Optional[ExponentialMovingAverage] = None, # Not tested yet
    plotter: TrainingPlotter = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.ExponentialLR] = None,
    swa: Optional[int] = None, # Stochastic Weight Averaging [swacontainer] # Not tested yet
):
    """
    This function encapsulates all training steps to streamline the 'run_training' process.

    Performs the following actions:
    - Updates model weights using the optimizer.
    - Logs training metrics via the logger.
    - Updates and saves checkpoints for future use and monitoring.
    """

    # Unpack config
    eval_interval=config.eval_interval
    max_num_epochs=config.max_num_epochs
    patience=config.patience
    save_all_checkpoints=config.save_all_checkpoints
    max_grad_norm=config.clip_grad


    lowest_loss = np.inf
    valid_loss = np.inf


    patience_counter = 0
    keep_last = False

    if max_grad_norm is not None:
        logger.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logger.info("")
    logger.info("===========TRAINING===========")
    logger.info("Started training, reporting errors on validation set")
    logger.info("Loss metrics on validation set")
    epoch = start_epoch

    # Evalutation of the model before the training
    valid_loss, eval_metrics = evaluate(
        model = model,
        loss_fn = loss_fn,
        data_loader = valid_loader,
        device = device,
        metrics_to_print = config.metrics_to_print,
    )
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch

    print(" loss before training: ", valid_loss)
    print()
    print()

    # Update ploting list in result data class, feel free to add any things you want to plot
    Global_class_results.plotting_datas.append(eval_metrics)

    # Write Metrics in Result file
    results_logger.log(eval_metrics)

    # Write in Logger
    logger.info(f"Metrics on evaluation before training:")
    logger.info(eval_metrics)
    

    while epoch < max_num_epochs:

        ######################### Training #######################

        print("")
        print(f"\033[34m Training epoch {epoch+1}/{max_num_epochs} \033[0m")
        print("")

        # LR scheduler
        if epoch > start_epoch:
            lr_scheduler.step(metrics=valid_loss)  # Can break if exponential LR, TODO fix that

        # SWA has been disabeled for now
        #else:
        #    if swa_start:
        #        logger.info("Changing loss based on Stage Two Weights")
        #        lowest_loss = np.inf
        #        swa_start = False
        #        keep_last = True
        #    loss_fn = swa.loss_fn
        #    swa.model.update_parameters(model)
        #    if epoch > start_epoch:
        #        swa.scheduler.step()

        # ScheduleFree
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()

        # 1st epoch is 1 not 0 (0 is pretraining eval)
        epoch += 1

        # LBFGS # Not tested yet
        if isinstance(optimizer, LBFGS):

            # Learn and metrics output
            _, opt_metrics = take_step_lbfgs(
                model=model,
                loss_fn=loss_fn,
                data_loader=train_loader,
                optimizer=optimizer,
                ema=ema,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch

            # Save metrics
            Global_class_results.plotting_datas.append(opt_metrics)
            if results_logger is not None:
                results_logger.log(opt_metrics)

        else: # Not LBFGS
            for batch in tqdm(train_loader):

                # Learn and metrics output
                _, opt_metrics = take_step(
                    model=model,
                    loss_fn=loss_fn,
                    batch=batch,
                    optimizer=optimizer,
                    ema=ema,
                    max_grad_norm=max_grad_norm,
                    device=device,
                )
                
                opt_metrics["mode"] = "opt"
                opt_metrics["epoch"] = epoch

                # Save metrics
                Global_class_results.plotting_datas.append(opt_metrics)
                if results_logger is not None:
                    results_logger.log(opt_metrics)
                



        ######################### Evaluate #######################

        if epoch % eval_interval == 0:

            model_to_evaluate = model

            #  For ema
            param_context = (ema.average_parameters() if ema is not None else nullcontext())


            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()


            with param_context: # For ema contexte
                valid_loss, eval_metrics = evaluate(
                    model=model_to_evaluate,
                    loss_fn=loss_fn,
                    data_loader=valid_loader,
                    device=device,
                    metrics_to_print=config.metrics_to_print,
                )
                eval_metrics["mode"] = "eval"
                eval_metrics["epoch"] = epoch


                if results_logger is not None:
                    results_logger.log(eval_metrics)


                #plotter
                if plotter and epoch % plotter.plot_frequency == 0:
                    try:
                        plotter.plot(epoch, model_to_evaluate)
                    except Exception as e:
                        logger.error("Plotting failed with an exception:")
                        logger.error(traceback.format_exc())

                valid_loss = (valid_loss)


            if valid_loss >= lowest_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    if swa is not None and epoch < swa.start:
                        logger.info(
                            f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                        )
                        epoch = swa.start
                    else:
                        logger.info(
                            f"Stopping optimization after {patience_counter} epochs without improvement"
                        )
                        break
                     
                if patience_counter == 1: # Save checkpoint if no improvement
                    param_context = (ema.average_parameters() if ema is not None else nullcontext())
                    with param_context:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=True,
                        )
            else:
                lowest_loss = valid_loss
                patience_counter = 0
                param_context = (
                    ema.average_parameters() if ema is not None else nullcontext()
                )
                with param_context:
                    checkpoint_handler.save(
                        state=CheckpointState(model, optimizer, lr_scheduler),
                        epochs=epoch,
                        keep_last=keep_last,
                    )
                    keep_last = False or save_all_checkpoints


    logger.info("Training complete")
    return epoch

