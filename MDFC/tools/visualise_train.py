###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/cli/visualise_train.py
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
#
###########################################################################################

import os

import logging

from typing import Dict, List, Optional
from .config_tools import config_class_training
from .torch_geometric.dataloader import DataLoader
from dataclasses import dataclass
from .torch_tools import to_numpy


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchmetrics import Metric
from prettytable import PrettyTable
 
logger = logging.getLogger("Main_Logger")
plt.rcParams.update({"font.size": 8})
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

colors = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#7f7f7f",  # middle gray
    "#2ca02c",  # cooked asparagus green
    "#ff7f0e",  # safety orange
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

Metrics_name_for_plots = { # Name of the metrics used in the plots
    "percentile": "Percentile 95",
    "relativeRMSE": "Relative RMSE",
    "RMSE": "RMSE",
    "relativeMAE": "Relative MAE",
    "MAE": "MAE",
}

# See documentation for more details on error types
# New metrics can be added by adding a new tuple in the 2 lists of a key of the dictionary (also need in the error function (todo: be more specific))


@dataclass
class Results_dataclass:
    """
    Dataclass to store results for each epoch.
    this class will be accessible from anywhere in the code.
    """
    plotting_datas: list[Dict] #int: epoch, list: metrics
    # Please feel free to add more attributes to this class as needed

Global_class_results = Results_dataclass([],) # "from visualise_train import Global_class_results" to access class


class TrainingPlotter:
    def __init__(self,
        plot_dir  : str,
        config: config_class_training,
        metrics_to_print: list,
        train_valid_data: Dict[str, DataLoader],
        test_data: Dict,
        tag: str,
        device: str,
        plot_frequency: int,
        swa_start: Optional[int] = None,
    ):
        self.plot_dir   = plot_dir  
        self.tag = tag
        self.config = config
        self.metrics_to_print = metrics_to_print
        self.train_valid_data = train_valid_data
        self.test_data = test_data
        self.device = device
        self.plot_frequency = plot_frequency
        self.swa_start = swa_start

    def plot(self, 
        model_epoch: str, 
        model: torch.nn.Module, 
    ) -> None:

        train_valid_inference_dict = model_inference(
            self.train_valid_data,
            model,
            self.device,
        )

        if len(self.test_data) != 0:
            test_inference_dict = model_inference(
                self.test_data,
                model, 
                self.device, 
            )
        else:
            test_inference_dict = {}

        # PD_data have {loss, time, epoch, mode} as columns
        PD_data = pd.DataFrame(results for results in Global_class_results.plotting_datas)


        # Creating general fig properties
        fig = plt.figure(layout="constrained", figsize=(10, 6))
        fig.suptitle(f"Model loaded from epoch {model_epoch}", fontsize=16)

        subfigs = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.05)
        axsTop = subfigs[0].subplots(1, 2, sharey=False)
        axsBottom = subfigs[1].subplots(1, 2, sharey=False)
        
        # Plot Top panel of the fig
        plot_top(
            axes = axsTop, 
            PD_data = PD_data, 
            model_epoch = model_epoch, 
            metrics_to_print = self.metrics_to_print
        )
        # Plot Bot panel of the fig
        plot_bot(
            axsBottom, 
            train_valid_inference_dict, 
            test_inference_dict, 
            self.metrics_to_print, 
            model_epoch, 
            PD_data
        )

        
        if self.swa_start is not None:
            # Add vertical lines to both axes
            for ax in axsTop:
                ax.axvline(
                    self.swa_start,
                    color="black",
                    linestyle="dashed",
                    linewidth=1,
                    alpha=0.6,
                    label="Stage Two Starts",
                )


        axsTop[0].legend(loc="best")
        # Save the figure using the appropriate stage in the filename
        filename = f"{self.plot_dir}/{self.tag}_epoch_{model_epoch}_plot.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_top(
    axes: np.ndarray, # 2 axes, one for loss and one for the other metrics
    PD_data: pd.DataFrame, 
    model_epoch: str, # Lasrt epoch of the model
    metrics_to_print: List[str] # Labels for the metrics to be plotted
) -> None:

    valid_data = (
        PD_data[PD_data["mode"] == "eval"] # Diferentiate between training and validation data (flaged as "opt" and "eval" in "mode" column in the jsonl file)
        .groupby(["mode", "epoch"]) # Only "eval" mode is now keept, this line is used to group the data by epoch
        .agg(["mean", "std"]) # We calculate the mean and std of eatch epoch
        .reset_index() # Rearange the data per epoch
    )

    train_data = (
        PD_data[PD_data["mode"] == "opt"] # Diferentiate between training and validation data (flaged as "opt" and "eval" in "mode" column in the jsonl file)
        .groupby(["mode", "epoch"]) # Only "eval" mode is now keept, this line is used to group the data by epoch
        .agg(["mean", "std"]) # We calculate the mean and std of eatch epoch
        .reset_index() # Rearange the data per epoch
    )

    ################## Plot loss/Epoch ######################

    # Loss axe
    ax_Left = axes[0]

    # Plot 2 graphs 
    ax_Left.plot(train_data["epoch"], train_data["loss"]["mean"], color=colors[1], linewidth=1, label="Training Loss")
    ax_Left.plot(valid_data["epoch"], valid_data["loss"]["mean"], color=colors[0], linewidth=1, label="Validation Loss")
    ax_Left.set_ylabel("Loss")
    ax_Left.set_yscale("log")

    # Add Loaded Model line
    ax_Left.axvline(
        model_epoch,
        color="black",
        linestyle="solid",
        linewidth=1,
        alpha=0.8,
        label="Loaded Model",
    )

    # Set x label and drid
    ax_Left.set_xlabel("Epoch")
    ax_Left.grid(True, linestyle="--", alpha=0.5)

    ################## Plot %_metrics/Epoch ######################

    # %_metrics axe
    ax_Right = axes[1]

    # "relative" mean %
    for i, label in enumerate(metrics_to_print):
        if "relative" in label:

            # Feel free to customize your colors
            color = colors[(i + 3)]

            # Plot metrics
            ax_Right.plot(
                valid_data["epoch"],
                valid_data[label]["mean"],
                color=color,
                label=label,
                linewidth=1,
            )

            # Settings
            ax_Right.set_yscale("log")
            ax_Right.set_ylabel(Metrics_name_for_plots[label], color=color)
            ax_Right.tick_params(axis="y", colors="black")
            ax_Right.legend()

    # Add Loaded Model line
    ax_Right.axvline(
        model_epoch,
        color="black",
        linestyle="solid",
        linewidth=1,
        alpha=0.8,
        label="Loaded Model",
    )
    ax_Right.set_xlabel("Epoch")
    ax_Right.grid(True, linestyle="--", alpha=0.5)



def plot_bot(
    axes: np.ndarray,
    train_valid_inference_dict: dict,
    test_inference_dict: dict,
    metrics_to_print: list,
    model_epoch: str, 
    PD_data: pd.DataFrame
) -> None:

    ################## Plot metrics/Epoch ######################

    valid_data = (
        PD_data[PD_data["mode"] == "eval"] # Diferentiate between training and validation data (flaged as "opt" and "eval" in "mode" column in the jsonl file)
        .groupby(["mode", "epoch"]) # Only "eval" mode is now keept, this line is used to group the data by epoch
        .agg(["mean", "std"]) # We calculate the mean and std of eatch epoch
        .reset_index() # Rearange the data per epoch
    )

    # metrics axe
    Left_ax = axes[0]

    # Name and grid
    Left_ax.set_xlabel("Epoch")
    Left_ax.grid(True, linestyle="--", alpha=0.5)

    # Same as top Right
    for i, label in enumerate(metrics_to_print):
        if "relative" not in label:
            color = colors[(i + 3)]

            Left_ax.plot(
                valid_data["epoch"],
                valid_data[label]["mean"],
                color=color,
                label=label,
                linewidth=1,
            )
            Left_ax.set_yscale("log")
            Left_ax.set_ylabel(Metrics_name_for_plots[label], color=color)
            Left_ax.tick_params(axis="y", colors="black")
            Left_ax.legend()
    Left_ax.axvline(
        model_epoch,
        color="black",
        linestyle="solid",
        linewidth=1,
        alpha=0.8,
        label="Loaded Model",
    )
    Left_ax.set_xlabel("Epoch")
    Left_ax.grid(True, linestyle="--", alpha=0.5)

    ################## Plot last epoch metrics ######################

    axsBottomR = axes[1]

    # Plot train/valid data (each entry keeps its own name)
    for name, inference in train_valid_inference_dict.items():
        # Separate train and valid visualisation
        if "train" in name:
            fixed_color_train_valid = colors[(i + 1)]
            marker = "x"
        else:
            fixed_color_train_valid = colors[(i + 2)]
            marker = "+"

        # Print points 
        scatter = axsBottomR.scatter(
            inference["forces"]["reference"],
            inference["forces"]["predicted"],
            marker=marker,
            color=fixed_color_train_valid,
            label=name,
        )



    fixed_color_test = colors[2]  # Color for test dataset
    # Plot test data (single legend entry)
    for name, inference in test_inference_dict.items():

        # Initialize scatter to None to avoid possibly used before assignment
        scatter = axsBottomR.scatter(
            inference["forces"]["reference"],
            inference["forces"]["predicted"],
            marker="o",
            color=fixed_color_test,
            label="Test",
        )

    # Add diagonal line for guide
    min_val = min(axsBottomR.get_xlim()[0], axsBottomR.get_ylim()[0])
    max_val = max(axsBottomR.get_xlim()[1], axsBottomR.get_ylim()[1])
    axsBottomR.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="black",
        alpha=0.7,
    )


    axsBottomR.set_xlabel(f"Reference ")
    axsBottomR.set_ylabel(f"Predicted ")
    axsBottomR.grid(True, linestyle="--", alpha=0.5)


def model_inference(
    all_data_loaders: dict, # all_data_loaders is a dict of data loaders. !! In this version of the code, we only have one data loader in the dict !!
    model: torch.nn.Module,
    device: str,
):
    """
    Run inference on the model using the provided data loaders list.
    Args:
        all_data_loaders (dict): Dictionary of data loaders.
        model (torch.nn.Module)
        device (str)
    """

    for param in model.parameters():
        param.requires_grad = False

    results_dict = {}

    for name in all_data_loaders: # all_data_loaders is a dict of data loaders. !! In this version of the code, we only have one data loader in the dict !!
        data_loader = all_data_loaders[name]
        logger.debug(f"Running inference on {name} dataset")
        scatter_metric = InferenceMetric().to(device)

        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=False,
            )

            results = scatter_metric(batch, output)

        results = scatter_metric.compute()
        results_dict[name] = results
        scatter_metric.reset()

        del data_loader

    for param in model.parameters():
        param.requires_grad = True

    return results_dict


class InferenceMetric(Metric):
    """Metric class for collecting reference and predicted values for scatterplot visualization."""

    def __init__(self):
        super().__init__()

        # Reference and predicted values
        self.add_state("ref_forces", default=[], dist_reduce_fx="cat")
        self.add_state("pred_forces", default=[], dist_reduce_fx="cat")
 

        # Store atom counts for each configuration
        self.add_state("atom_counts", default=[], dist_reduce_fx="cat")
        self.add_state("n_forces", default=torch.tensor(0.0), dist_reduce_fx="sum")


    def update(self, batch, output):  # pylint: disable=arguments-differ
        """Update metric states with new batch data."""

        # Calculate number of atoms per configuration
        atoms_per_config = batch.ptr[1:] - batch.ptr[:-1]
        self.atom_counts.append(atoms_per_config)

        # Forces
        self.n_forces += 1.0
        self.ref_forces.append(batch.snapshot_particules_force)
        self.pred_forces.append(output["forces"])


    def _process_data(self, ref_list, pred_list):
        # Handle different possible states of ref_list and pred_list in distributed mode

        # Check if this is a list type object
        if isinstance(ref_list, (list, tuple)):
            if len(ref_list) == 0:
                return None, None
            ref = torch.cat(ref_list).reshape(-1)
            pred = torch.cat(pred_list).reshape(-1)
        # Handle case where ref_list is already a tensor (happens after reset in distributed mode)
        elif isinstance(ref_list, torch.Tensor):
            ref = ref_list.reshape(-1)
            pred = pred_list.reshape(-1)
        # Handle other possible types
        else:
            return None, None
        return to_numpy(ref), to_numpy(pred)

    def compute(self):
        """Compute final results for scatterplot."""
        results = {}

        # Process forces
        ref_f, pred_f = self._process_data(self.ref_forces, self.pred_forces)
        results["forces"] = {
            "reference": ref_f,
            "predicted": pred_f,
        }

        return results

def print_comparaison_file(
    model: torch.nn.Module,
    tables_test: PrettyTable,
    data_loader: DataLoader,
    test_dir: str,
    device: torch.device,
):

    # For testing, desactivate requires_grad
    for param in model.parameters():
        param.requires_grad = False

    # Run the model
    for name, data in data_loader.items():
        for i, batch in enumerate(data):
            if i == 1:
                print("i == 1")
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(batch_dict, training=False)

        file_path = os.path.join(test_dir, name + "_results")
        file_path = os.path.expanduser(file_path)
        if os.path.exists(file_path):
            logger.warning(f"File {file_path} already exists. Overwriting.")
            os.remove(file_path)
        
        logger.info(f"Writing testing results to {file_path}")
        with open(file_path, "w") as file:
            file.write(f"snapshot total energy: {str(output['snapshots_energy'].detach().cpu().numpy())}\n \n")
            file.write(f"particuls predicted forces: \n \n {str(output['forces'].detach().cpu().numpy())} \n \n")
            file.write(f"particuls real forces: \n \n {str(batch_dict['snapshot_particules_force'].detach().cpu().numpy())}\n \n")
            file.write(str(tables_test[name]))



    # reactivate requires_grad
    for param in model.parameters():
        param.requires_grad = True


