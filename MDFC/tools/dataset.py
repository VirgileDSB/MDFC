###########################################################################################
#
# First Authors: Virgile de Saint Blanquat
#
###########################################################################################

from torch.utils.data import Dataset
import numpy as np
import os
import torch
from .torch_tools import to_one_hot
from .torch_geometric.data import Data
from .positions_to_edges import positions_to_edges
import glob


class CustomDataset(Dataset):
    """
    Custom dataset for reading molecular dynamics data from files.
    Use torch_geometric.data to handle graph data structures.
    """
    def __init__(self, 
        filepaths, 
        cutoff:float, 
        num_classes: int,
        default_dtype: str,
        F_mul: float,
    ):
        """
        init() will read the datas from a file and return a torch_geometric dataset
        """

        filepaths = glob.glob(os.path.expanduser(filepaths))
        datas = [] # Container of all datas of the file [N_snapshot, ]

        for filepath in filepaths: # Loop on all files
            with open(filepath, 'r') as file:


                while True: # Loop on all snapshots in 1 file

                    # Containers of 1 snapshot datas
                    one_snapshot_particules_position = []
                    one_snapshot_particules_type = []
                    one_snapshot_particules_force = []
                    self.shift_datas = []

                    try: 
                        n_part = int(file.readline()) # 1st line is the number of parts so an integer
                    except ValueError:
                        raise ValueError(f"problem with data file, please make sure that there is no empty lines at the end of the file")

                    box_size = np.float64(file.readline().split()) # 2nd line is box size

                    # For the number of parts, read datas
                    for _ in range(n_part): 
                        data = file.readline().split() # Read one line

                        particule_type = torch.tensor(ord(str(data[0])) - ord('a'), dtype=torch.int64) # Particule type converted from str to int64 (a=1, b=2, ...)

                        # Positions (modulo box size to avoids particles going out of the box sometimes wich is not a problem in MD but can be in ML)
                        p_x = np.mod(float(data[1]), box_size[0])
                        p_y = np.mod(float(data[2]), box_size[1])
                        p_z = np.mod(float(data[3]), box_size[2])
                        particule_position = np.array([p_x,p_y, p_z])

                        # Forces
                        f_x = float(data[4])*F_mul
                        f_y = float(data[5])*F_mul
                        f_z = float(data[6])*F_mul
                        particule_force = np.array([f_x,f_y, f_z])

                        # "one_snapshot" coresponf to one data
                        one_snapshot_particules_type.append(to_one_hot(particule_type, num_classes))
                        one_snapshot_particules_position.append(particule_position)
                        one_snapshot_particules_force.append(particule_force)

                    one_snapshot_particules_type = torch.stack(one_snapshot_particules_type)

                    ########## 1st layer of embedding wich is not IA related ##########

                    edge_index, shifts, unit_shifts = positions_to_edges(
                        positions=one_snapshot_particules_position, cutoff=cutoff, boxs_size=box_size, default_dtype=default_dtype
                    )

                    unit_shifts = torch.tensor(unit_shifts)
                    one_snapshot_particules_position = torch.tensor(np.array(one_snapshot_particules_position), dtype=getattr(torch, default_dtype))
                    one_snapshot_particules_force = torch.tensor(np.array(one_snapshot_particules_force), dtype=getattr(torch, default_dtype))
                    box_size = torch.tensor(box_size, dtype=getattr(torch, default_dtype))

                    # This is all the datas contained in the snapshot in a torch_geometric.data.Data object
                    one_snapshot_datas =  Data(
                        x = one_snapshot_particules_type, 
                        edge_index = edge_index, 
                        positions = one_snapshot_particules_position, 
                        shifts = shifts, 
                        unit_shifts = unit_shifts,
                        snapshot_particules_force = one_snapshot_particules_force,
                        box_size = box_size,
                        num_part = n_part,
                    )

                    # All snapshots list
                    datas.append(one_snapshot_datas)

                    if not file.readline():
                        break

        if len(datas) == 0:
            raise ValueError(f"problem with data file, no file found in {filepaths}")
        self.datas = datas
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        return self.datas[idx]
    

    
