###########################################################################################
#
# First Authors: Virgile de Saint Blanquat
#
###########################################################################################

import sys
from pathlib import Path
from .dataset import CustomDataset
import logging
from typing import Dict
import torch
from .config_tools import config_run_model

logger = logging.getLogger("Main_Logger")


def update_progress(progress, bar_length=40):
    """
    Update the progress bar in the terminal.
    """
    block = int(round(bar_length * progress))
    bar = '#' * block + '-' * (bar_length - block)
    text = f"\rProgression: [{bar}] {int(progress * 100)}%"
    sys.stdout.write(text)
    sys.stdout.flush()



def print_output_file(dataset: CustomDataset, output: Dict[str, torch.Tensor], config: config_run_model, tag: str) -> None:
    """
    Print a output file in a .sph like format 
    """

    # Create the output file name
    output_file_name = f"{tag}.sph"
    output_file_path = Path(config.results_path) / output_file_name

    output_file_path = Path(output_file_path).expanduser()
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    while output_file_path.exists(): # Be sure that there is no overlap with existing files
        logger.warning(f"Output file {output_file_path} already exists, creating a new one.")
        output_file_path = output_file_path.with_name(output_file_path.stem + "_2" + ".sph")
        logger.info(f"Output file will be saved as {output_file_path}")
        


    with open(output_file_path, 'w') as file:
        file.write(str(dataset[0].num_part) + "\n")  # 1st line is number of parts
        file.write(str(dataset[0].box_size[0].item()) + " " + str(dataset[0].box_size[1].item()) + " " + str(dataset[0].box_size[2].item()) + "\n")  # 2nd line is box size
        for i in range(dataset[0].num_part): 
            file.write(str(chr(ord('a') + int(dataset[0].x[i].item()) - 1)) + " ")  # Write particle type as a letter
            file.write(str(dataset[0].positions[i][0].item()) + " " + str(dataset[0].positions[i][1].item()) + " " + str(dataset[0].positions[i][2].item()) + " ")  # Write particle position
            file.write(str(output["forces"][i][0].item()/config.F_mul) + " " + str(output["forces"][i][1].item()/config.F_mul) + " " + str(output["forces"][i][2].item()/config.F_mul))  # Write particle force
            file.write("\n")

    logger.info(f"Output saved in {output_file_path}")

