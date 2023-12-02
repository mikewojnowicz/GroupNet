import numpy as np
import argparse
import os
import sys
import random
sys.path.append(os.getcwd())
import torch
import matplotlib.pyplot as plt

from groupnet.model.GroupNet_nba import GroupNet
from groupnet.compare.data_loader import DATA_DIR 
from groupnet.compare.examples import  get_start_and_stop_timestep_idxs_from_event_idx
from groupnet.compare.court import normalize_coords_from_meters, unnormalize_coords_to_meters

### 
# HELPERS    
###

def make_context_sets_in_meters(past_length) -> torch.Tensor:
    """
    context sets looks like past_trajs
    has shape:   (num_examples, num_players_plus_ball=11, past_length, dims_of_court=2)
    """

    """load stuff to make forecast"""
    coords_filepath = os.path.join(DATA_DIR, "player_coords_test__with_5_games.npy")
    example_stop_idxs_filepath = os.path.join(DATA_DIR, "example_stop_idxs_test__with_5_games.npy")
    random_context_times_filepath = os.path.join(DATA_DIR, "random_context_times.npy")

    xs_test = np.load(coords_filepath)
    example_end_times_test = np.load(example_stop_idxs_filepath)
    random_context_times = np.load(random_context_times_filepath)

    """ make forecasts"""
    event_idxs_to_analyze = [
        i for (i, random_context_time) in enumerate(random_context_times) if not np.isnan(random_context_time)
    ]

    num_examples = len(event_idxs_to_analyze)
    num_players = 10 
    num_players_plus_ball=11
    dims_of_court=2

    context_sets_normalized = np.zeros((num_examples, num_players_plus_ball, past_length, dims_of_court))

    for e, event_idx_to_analyze in enumerate(event_idxs_to_analyze):
        start_idx, stop_idx = get_start_and_stop_timestep_idxs_from_event_idx(
            example_end_times_test, event_idx_to_analyze
        )
        xs_test_example = xs_test[start_idx:stop_idx]
        T_context = int(random_context_times[event_idx_to_analyze])

        context_sets_normalized[e:,:num_players] = xs_test_example[T_context-past_length: T_context].swapaxes(0,1) # pre-swap shape (T,J,D); post-swap: (J,T,D)

    context_sets = unnormalize_coords_to_meters(context_sets_normalized)
    return torch.Tensor(context_sets)



### 
# MAIN
###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_games", type=int, default=None) # in [1,5,20]
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--traj_scale', type=int, default=1)
    parser.add_argument('--sample_k', type=int, default=20)
    parser.add_argument('--past_length', type=int, default=10)
    parser.add_argument('--future_length', type=int, default=30)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)


    """ seeds """
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    """ load model """
    MODEL_SAVE_DIR = f'saved_models/nba/{args.num_train_games}_train_games/'
    model_path = os.path.join(MODEL_SAVE_DIR, str(args.model_name)+'.p')
    print('load model from:', model_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    training_args = checkpoint['model_cfg']

    model = GroupNet(training_args,device)            
    model.set_device(device)
    model.eval()
    model.load_state_dict(checkpoint['model_dict'], strict=True)

    """ make forecasts """
    context_sets = make_context_sets_in_meters(args.past_length)
    num_examples,num_players_plus_ball, past_length, num_court_dims=np.shape(context_sets)
    data={"past_traj":  context_sets}
    with torch.no_grad():
        forecasts_in_meters_and_compressed = model.inference(data)
    num_forecasts=len(forecasts_in_meters_and_compressed)
    result = forecasts_in_meters_and_compressed.view(num_forecasts, num_examples, num_players_plus_ball, args.future_length, num_court_dims)
    forecasts_in_meters = result[:,:,:10].cpu().numpy()
    forecasts = normalize_coords_from_meters(forecasts_in_meters) # shape (S,E,J,T,D)
    forecasts_savepath=f"forecasts_{args.num_train_games}_train_games.npy"
    np.save(forecasts_savepath, forecasts)
    print(f"Wrote forecasts to {forecasts_savepath}.")