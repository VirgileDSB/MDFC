import logging
import os
from typing import Union
from typing import Dict, Any
import json
import torch
import numpy as np

logger = logging.getLogger("Main_Logger")

class UniversalEncoder(json.JSONEncoder):
    """
    Transform numpy and torch objects to JSON serializable types to be printed in a text file.
    """
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.cpu().detach().numpy()
        return json.JSONEncoder.default(self, o)

def setup_logger(
    level_print_terminal: Union[int, str],
    level_print_file: Union[int, str],
    tag: str,
    directory: str,
):
    """
    Setup the logger.
    """

    # Create a general logger object "Main_Logger"
    logger = logging.getLogger("Main_Logger")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Create formatters
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",)

    ### Set levels for terminal ###
    console_handler = logging.StreamHandler() # Basically, says that we want to print the logs in the terminal
    console_handler.setLevel(level_print_terminal) # Set the level of the console handler
    console_handler.setFormatter(formatter) # Set the formatter for the console handler
    logger.addHandler(console_handler) # Connect the console handler to the "Main_Logger" witch is called 'logger'

    ### Set levels for file ###
    
    # Create the directory and the log file
    main_log_path = os.path.join(directory, f"{tag}.log")
    print(f"Setting logger to {main_log_path}")
    if os.path.exists(main_log_path):
        print("")
        print(f"\033[31mA old logger file was found at {main_log_path}, delleting it\033[0m")
        print("")
        os.remove(main_log_path)

    # Setup the file handler
    file_handler = logging.FileHandler(main_log_path)
    file_handler.setLevel(level_print_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class MetricsLogger:
    def __init__(
            self, 
            directory: str, 
            tag: str
        ) -> None:
        self.directory = directory
        self.filename = tag + ".txt"
        self.path = os.path.join(self.directory, self.filename)
        print(f"Setting result file to {self.path}")
        if os.path.exists(self.path):
            logger.warning(f"A old result file was found at {self.path}, delleting it")
            os.remove(self.path)

    def log(
            self, 
            d: Dict[str, Any]
        ) -> None:
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write("\n")