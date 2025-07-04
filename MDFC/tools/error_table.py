###########################################################################################
#
# Script inspired by github.com/ACEsuit/tools/script_utils.py
#       First Authors: David Kovacs, Ilyes Batatia
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

import logging
import torch
from prettytable import PrettyTable
from ..model.evaluate import evaluate

logger = logging.getLogger("Main_Logger")

def create_error_table(
    metrics_to_print: list[str],
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: str,
) -> PrettyTable:
    

    # Setup table
    table = PrettyTable()

    # Evaluation
    _, metrics = evaluate(
        model,
        loss_fn=loss_fn,
        data_loader=data_loader,
        device=device,
        metrics_to_print=metrics_to_print
    )


    if "percentile" in metrics_to_print:
        table.add_column("Percentile 95", [metrics['percentile']])

    if "relativeRMSE" in metrics_to_print:
        table.add_column("relative F RMSE %", [metrics['relativeRMSE']])

    if "RMSE" in metrics_to_print:
        table.add_column("RMSE F / atom", [metrics['RMSE']])

    if "relativeMAE" in metrics_to_print:
        table.add_column("relative F RMSE %", [metrics['relativeMAE']])

    if "MAE" in metrics_to_print:
        table.add_column("MAE F / atom", [metrics['MAE']])

    return table