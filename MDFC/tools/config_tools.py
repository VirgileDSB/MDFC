###########################################################################################
#
# First Authors: Virgile de Saint Blanquat
#
###########################################################################################


import os
from dataclasses import dataclass
from e3nn import o3
from typing import Optional, Literal, Union
import cuequivariance as cue

training_default_config = {
    "seed": 123 , "device": "cuda", "default_dtype": "float64", "name": "training", "cutt_off": 2.0, "num_elements": 1, "F_mul": 1, #general
    "work_dir": "." , "log_dir": None, "model_dir": None, "checkpoints_dir": None, "results_dir": None, "plot_dir": None, #dir
    "file_log_level": "DEBUG", "terminal_log_level": "WARNING", "plot": True, "plot_frequency": 0, "metrics_to_print": ["RMSE", "percentile", "relativeRMSE", "relativeMAE", "MAE"], #infos
    "num_radial_basis": 8, "num_polynomial_cutoff": 5, "max_ell":3, "num_channels": 128, #Embedding
    "interaction": "RealAgnosticResidualInteractionBlock", "interaction_first": "RealAgnosticResidualInteractionBlock", "num_interactions":2, "correlation": [3], #interactions
    "MLP_irreps": "16x0e", "radial_MLP": [64, 64, 64], "hidden_irreps": None, "max_L": 1, #hidden layers
    "gate": "silu", "loss_fct": "projected_mean_squared_error", #output
    "layout": "mul_ir", #cuet
    "train_file": None, "valid_file": None, "valid_fraction": None, "test_dir": None, #datas
    "num_workers": 0, "pin_memory": True,"test_infos": False, #data loading
    "optimizer": "adam", "beta": 0.9, "batch_size": 3, "valid_batch_size":10, "lr":0.01, "weight_decay": 5e-7, "clip_grad": 10, "amsgrad": False, "avg_num_neighbors": 1, #optimization
    "scheduler": "ReduceLROnPlateau", "lr_factor": 0.8, "scheduler_patience": 50, "lr_scheduler_gamma": 0.9993, #scheduler
    "ema": False, "ema_decay": 0.99, #Exponential Moving Average
    "max_num_epochs": 2, "patience": 2048, #Epoch
    "eval_interval": 1, #eval
    "keep_checkpoints": False, "save_all_checkpoints": False,  "restart_latest": False, #Checkpoints
}

model_keys = [   
    "num_interactions",
    "cutt_off",
    "num_channels",
    "num_radial_basis",
    "num_polynomial_cutoff",
    "max_ell",
    "radial_MLP",
    "hidden_irreps",
    "avg_num_neighbors",
    "MLP_irreps",
    "gate",
    "layout",
    "interaction",
    "interaction_first",
    "num_elements",
    "correlation"
]

use_model_default_config = {
    "cutt_off": 2.0, "num_elements": 1, "num_channels": 128, "num_radial_basis": 8, "num_polynomial_cutoff": 5, "max_ell":3, "default_dtype": "float64", "F_mul": 1, "max_L": 1,
    "interaction": "RealAgnosticResidualInteractionBlock", "interaction_first": "RealAgnosticResidualInteractionBlock", "num_interactions":2, "correlation": [3], "gate": "silu",
    "MLP_irreps": "16x0e", "radial_MLP": [64, 64, 64], "hidden_irreps": None, "layout": "mul_ir", "avg_num_neighbors":1,
    "device": "cuda", "num_workers": 0, "pin_memory": True,
    "file_path": None,  "results_path": None, "model_path": ".",
    "log_dir": ".", "file_log_level": "DEBUG"
}

def update_config_training(config: dict) -> dict:
    """
    Update the default configuration with the provided configuration.
    Only for the training part of the code.
    """

    config_new = training_default_config.copy() 
    config_new.update(config)

    unknow_keys = set(config_new) - set(training_default_config)
    
    if unknow_keys:
        print(f"Error : unknow config key : {', '.join(unknow_keys)}") 

    # Use work_dir for all other directories as well, unless they were specified by the user
    if config_new["log_dir"] is None:
        config_new["log_dir"] = os.path.join(config_new["work_dir"], "logs")
    if config_new["model_dir"] is None:
        config_new["model_dir"] = config_new["work_dir"]
    if config_new["checkpoints_dir"] is None:
        config_new["checkpoints_dir"] = os.path.join(config_new["work_dir"], "checkpoints")
    if config_new["results_dir"] is None:
        config_new["results_dir"] = os.path.join(config_new["work_dir"], "results")
    if config_new["plot_dir"] is None and config_new["plot"]:
        config_new["plot_dir"] = os.path.join(config_new["work_dir"], "plots")
        os.makedirs(config_new["plot_dir"], exist_ok=True)

    assert config_new["max_L"] >= 0, "max_L must be non-negative integer"
    config_new["hidden_irreps"] = cue.Irreps( "O3", (config_new["num_channels"] * o3.Irreps.spherical_harmonics(config_new["max_L"])).sort().irreps.simplify())
    config_new["MLP_irreps"] = cue.Irreps( "O3", config_new["MLP_irreps"])

    return config_new, {k: config_new[k] for k in model_keys}, unknow_keys

def update_config_use_model(config: dict) -> dict:
    """
    Update the default configuration with the provided configuration.
    This is used to run the model after training.
    """


    config_new = use_model_default_config.copy() 
    config_new.update(config)

    unknow_keys = set(config_new) - set(use_model_default_config)
    
    if unknow_keys:
        print(f"Error : unknow config key : {', '.join(unknow_keys)}") 

    assert config_new["max_ell"] >= 0, "max_ell must be non-negative integer"
    config_new["hidden_irreps"] = cue.Irreps( "O3", (config_new["num_channels"] * o3.Irreps.spherical_harmonics(config_new["max_L"])).sort().irreps.simplify())
    config_new["MLP_irreps"] = cue.Irreps( "O3", config_new["MLP_irreps"])

    return config_new, {k: config_new[k] for k in model_keys}, unknow_keys


@dataclass
class config_class_training:
    r"""
    eatch elements in config dict is transform as a data in Config
    Only for the training part of the code.
    """
    seed: int
    device: Literal["cpu", "cuda", "mps", "xpu"]
    default_dtype: Literal["float32", "float64"]
    name: str
    cutt_off: float
    layout: str
    work_dir: str 
    amsgrad : bool
    num_elements: int
    avg_num_neighbors : float
    log_dir: str
    model_dir: str
    test_infos: bool
    checkpoints_dir: str
    file_log_level: str
    F_mul: float
    terminal_log_level: str
    hidden_irreps: cue.Irreps
    plot: bool
    plot_frequency: int
    metrics_to_print: list
    num_radial_basis: int 
    loss_fct: Literal[
            "projected_mean_squared_error",
            "euclidian_mean_squared_error",
        ] 
    num_polynomial_cutoff: int
    max_ell: int
    interaction: Literal[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticAttResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ] 
    interaction_first: Literal[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticAttResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ] 
    num_interactions:int
    correlation: Union[int, list[int]] #list/int of 1 int if all corelation the same for num_interactions eslse list of num_interactions int
    MLP_irreps: cue.Irreps
    radial_MLP: list
    num_workers: int
    pin_memory: bool
    optimizer: Literal["adam", "adamw", "schedulefree"]
    beta: float
    batch_size: int
    valid_batch_size:int
    lr:float
    weight_decay: float
    lr_factor: float
    clip_grad: int
    scheduler: str
    scheduler_patience: int
    lr_scheduler_gamma: float
    ema: bool
    ema_decay: float
    max_num_epochs: int
    patience: int
    eval_interval: int
    keep_checkpoints: bool
    save_all_checkpoints: bool
    restart_latest: bool 
    num_channels: int
    max_L: int
    plot_dir: Optional[str] = None
    gate: Optional[Literal["silu", "tanh", "abs"]] = None
    train_file: Optional[str] = None
    valid_file: Optional[str] = None
    valid_fraction: Optional[str] = None
    test_dir: Optional[str] = None
    results_dir: Optional[str] = None

@dataclass
class config_load_model:
    r"""
    All configs that are mandatory to run the model
    This class is shared between all run model methods
    """
    num_interactions: int
    cutt_off: float
    num_elements: int
    num_channels: int
    avg_num_neighbors: float
    num_radial_basis: int
    num_polynomial_cutoff: int
    max_ell: int
    radial_MLP: list
    hidden_irreps: cue.Irreps
    gate: str
    MLP_irreps: cue.Irreps
    layout: str
    interaction: Literal[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticAttResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ]
    interaction_first: Literal[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticAttResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ]
    correlation: Union[int, list[int]]

@dataclass
class config_run_model:
    r"""
    All configs that are mandatory to run the model (not training)
    """
    device: Literal["cpu", "cuda", "mps", "xpu"]
    num_workers: int
    pin_memory: bool
    log_dir: str
    max_L: int
    file_log_level: str
    model_path: str
    default_dtype: Literal["float32", "float64"]
    F_mul: float
    file_path: Optional[str] = None
    results_path: Optional[str] = None