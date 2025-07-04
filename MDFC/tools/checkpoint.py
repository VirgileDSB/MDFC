###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/tools/checkpoint.py
#       First Authors: Gregor Simm
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MDFC model and comments: Virgile de Saint Blanquat
###########################################################################################

import dataclasses
import logging
import os
import re
from typing import Dict, List, Optional
import torch
from .torch_tools import TensorDict

logger = logging.getLogger("Main_Logger")

Checkpoint = Dict[str, TensorDict]

@dataclasses.dataclass
class CheckpointState:
    """
    Data class that contain all the inforamtions about the model optimisation 
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR

@dataclasses.dataclass
class CheckpointPathInfo:
    """
    Data class for checkpoints file name infos
    """
    path: str
    tag: str
    epochs: int
    swa: bool


def load_checkpoint(
    state: CheckpointState, 
    checkpoint: Checkpoint, 
    strict: bool
) -> None:
    """
    Load the checkpoint "checkpoint" into local model
    """
    state.model.load_state_dict(checkpoint["model"], strict=strict) 
    state.optimizer.load_state_dict(checkpoint["optimizer"])
    state.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


class CheckpointHandler:
    """Class to handle saving and loading checkpoints."""
    def __init__(self, 
        directory: str,
        tag: str,
        keep: bool,
        swa_start: Optional[int] = None,
    ) -> None:
        """
        Initialize the CheckpointHandler with the directory to save checkpoints and a tag for the checkpoints names.
        """
        self.directory = directory # config.checkpoints_dir
        self.tag = tag
        self.keep = keep # Keep every checkpoints?
        self.old_path: Optional[str] = None # Path of the last checkpoint (for now not existing yet so set to None)
        self.swa_start = swa_start

        #strings for name
        self._epochs_string = "_epoch-" 
        self._filename_extension = "pt"

    def _get_checkpoint_filename(self, epochs: int, swa_start: Optional[int]) -> str:

        if swa_start is not None and epochs >= swa_start:
            return (self.tag + self._epochs_string + str(epochs) + "_swa"+ "." + self._filename_extension)
        
        return (self.tag + self._epochs_string + str(epochs) + "." + self._filename_extension)

    def _list_file_paths(self) -> List[str]:
        if not os.path.isdir(self.directory):
            return []
        all_paths = [
            os.path.join(self.directory, f) for f in os.listdir(self.directory)
        ]
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_checkpoint_path(self, path: str) -> Optional[CheckpointPathInfo]:
        """
        Get the tag and epochs from the checkpoint path name.
        """

        filename = os.path.basename(path)

        # Check if the file have a extractable name and extract the tag and epochs from the filename
        regex = re.compile(rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)\.{self._filename_extension}$")
        regex2 = re.compile(rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)_swa\.{self._filename_extension}$")
        match = regex.match(filename)
        match2 = regex2.match(filename)


        swa = False
        if not match:
            if not match2:
                return None # no match
            match = match2 # filename is a SWA checkpoint
            swa = True

        return CheckpointPathInfo(
            path=path,
            tag=match.group("tag"),
            epochs=int(match.group("epochs")),
            swa=swa,
        )

    def _get_latest_checkpoint_path(self, swa) -> Optional[str]:

        #create a list of all checkpoint paths with the same tag as the current one
        all_file_paths = self._list_file_paths()
        checkpoint_info_list = [self._parse_checkpoint_path(path) for path in all_file_paths]
        selected_checkpoint_info_list = [info for info in checkpoint_info_list if info and info.tag == self.tag]


        if len(selected_checkpoint_info_list) == 0:
            logger.warning(f"Cannot find checkpoint with tag '{self.tag}' in '{self.directory}'")
            return None


        selected_checkpoint_info_list_swa = []
        selected_checkpoint_info_list_no_swa = []


        for checkpoint in selected_checkpoint_info_list:
            if checkpoint.swa:
                selected_checkpoint_info_list_swa.append(checkpoint)
            else:
                selected_checkpoint_info_list_no_swa.append(checkpoint)
        if swa:
            try:
                latest_checkpoint_info = max(selected_checkpoint_info_list_swa, key=lambda info: info.epochs)
            except ValueError:
                logger.warning("No SWA checkpoint found, while SWA is enabled. Compare the swa_start parameter and the latest checkpoint.")
        else:
            latest_checkpoint_info = max(selected_checkpoint_info_list_no_swa, key=lambda info: info.epochs)
        return latest_checkpoint_info.path

    def save(self, 
        state: CheckpointState, 
        epochs: int, 
        keep_last: bool
    ) -> None:
        
        checkpoint = {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "lr_scheduler": state.lr_scheduler.state_dict(),
        }
        
        if not self.keep and self.old_path and not keep_last:
            logger.debug(f"Deleting old checkpoint file: {self.old_path}")
            os.remove(self.old_path)

        filename = self._get_checkpoint_filename(epochs, self.swa_start)
        path = os.path.join(self.directory, filename)
        logger.debug(f"Saving checkpoint: {path}")
        os.makedirs(self.directory, exist_ok=True)
        torch.save(obj=checkpoint, f=path)
        self.old_path = path

    def load_latest(
        self,
        state: CheckpointState,
        swa: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ) -> Optional[int]:
        """
        Find latest checkpoint and load it.
        """
        latest_path = self._get_latest_checkpoint_path(swa=swa)
        if latest_path is None:
            return None
        
        epochs = self.load(state = state, path = latest_path, device=device)

        return epochs

    def load(
        self,
        state: CheckpointState,
        path: str,
        device: Optional[torch.device] = None,
    ) -> int:
        """
        Load 'path' checkpoint.
        """
        checkpoint_info = self._parse_checkpoint_path(path)

        if checkpoint_info is None:
            raise RuntimeError(f"Cannot find path '{path}'")
        logger.info(f"Loading checkpoint: {checkpoint_info.path}")

        checkpoint = torch.load(f=checkpoint_info.path, map_location=device, weights_only=False) # Setting weights_only = True may be better, to test
        epochs = checkpoint_info.epochs

        load_checkpoint(state=state, checkpoint=checkpoint, strict=False)

        return epochs