
import os 
import numpy as np
import random 

import torch
from torch.utils.data import Dataset, DataLoader

from groupnet.compare.court import unnormalize_coords_to_meters

class BasketballDataset(Dataset):
    """
    Create a dataset which has __getitem__ defined to give a batch in the form expected by GroupNet

    1) Their model trains on meters, not coordinates normalized to the unit square.
    2) Their model trains on minibatches. 
    3) Their model expects to recieve a ball trajectory. 
    4) Their model requires a different data representation 
        4a) separate objects for past and future
        4b) torch tensors rather than numpy arrays with permuted dimensions
        ...etc.
    
    However, we deliver data in a way that matches experiments for the HSRDM paper:

    1) Their model expects to observe a ball trajectory, but our dataset doesn't have that.  
        We handle this by feeding their model a constant value (namely, 0) for the ball 
        coordinates (which is the 11th location in one-indexing).

    2) We feed their model minibatches that never straddle an example boundary.
    """
    def __init__(self, coords, example_stop_idxs, past_length, future_length, batch_size):
        self.coords = coords
        self.example_stop_idxs = example_stop_idxs
        self.past_length = past_length
        self.future_length = future_length
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        """
        Returns:
            Dict with keys 
                dict_keys(['past_traj', 'future_traj', 'seq'])
            such that
                result["seq"]="nba"
                result["past_traj"] is torch.Tensor with shape 
                    (batch_size, num_players_plus_ball=11, past_length, dims_of_court=2)
                result["future_traj"]  torch.Tensor with shape 
                    (batch_size, num_players_plus_ball=11, future_length, dims_of_court=2)
        """
        T = len(self.coords)
        num_players_plus_ball = 11
        court_dims = 2
        past_traj = torch.zeros((num_players_plus_ball, self.past_length, court_dims))
        future_traj = torch.zeros((num_players_plus_ball, self.future_length, court_dims))

        next_example_stop_idx, t_end = -np.inf, np.inf
        while next_example_stop_idx < t_end - 1:
            t_start = random.randint(0, T - 1)
            next_example_stop_idx = next((item for item in self.example_stop_idxs if item > t_start), None)
            t_end_plus_one = t_start + self.past_length + self.future_length
            t_end = t_end_plus_one - 1

        past_coords = self.coords[t_start:t_start + self.past_length]
        future_coords = self.coords[t_start + self.past_length:t_start + self.past_length + self.future_length]
        
        past_traj[:10] = torch.Tensor(unnormalize_coords_to_meters(past_coords)).permute(1, 0, 2)
        future_traj[:10] = torch.Tensor(unnormalize_coords_to_meters(future_coords)).permute(1, 0, 2)

        result = {'past_traj': past_traj,
                  'future_traj': future_traj,
                  'seq': 'nba'}

        return result

def make_data_loader(
    data_type : str, past_length: int = 10, future_length: int = 15, batch_size :int = 32
) -> DataLoader:
    
    """
    Arguments:
        data_type: str, in ["train_1", "train_5", "train_20", "test"]
    """


    # Example usage
    DATA_DIR = "data/basketball/baller2vec_format/processed/"

    if data_type == "train_1":
        coords_filepath = os.path.join(DATA_DIR, "player_coords_train__with_1_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_train__with_1_games.npy")
    elif data_type == "train_5":
        coords_filepath = os.path.join(DATA_DIR, "player_coords_train__with_5_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_train__with_5_games.npy")
    elif data_type == "train_20":
        coords_filepath = os.path.join(DATA_DIR, "player_coords_train__with_20_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_train__with_20_games.npy")
    elif data_type == "test":
        coords_filepath = os.path.join(DATA_DIR, "player_coords_test__with_5_games.npy")
        example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_test__with_5_games.npy")
    else: 
        raise ValueError(f"I don't understand data_type {data_type}, which should tell me whether to train or test"
                         f"and also the training size.  See function docstring.")  

    coords = np.load(coords_filepath)
    example_stop_idxs = np.load(example_stop_idxs_filepath)

    # Create a dataset which has __getitem__ defined to give a batch in the form expected by GroupNet
    basketball_dataset = BasketballDataset(coords, example_stop_idxs, past_length, future_length, batch_size)
    return DataLoader(basketball_dataset)


