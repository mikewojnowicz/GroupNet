    
import argparse
import numpy as np 

from groupnet.compare.train import train 

def test_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_games", type=int, default=1) 
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--its_per_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    # Their default for --past_length was 5.   
    # But their sampling rate is half of ours (they use 2.5 Hz vs our 5 Hz);
    # They say ["We predict the future 10 timestamps (4.0s) based on the historical 5 timestamps (2.0s)"]  
    # Hence, we use the sample past window size (2.0s), but adjust the number of timesteps to account for
    # sampling rate differneces. 
    parser.add_argument('--past_length', type=int, default=10) 
    # Their default for --future_length was 10.   
    # But their sampling rate is half of ours (they use 2.5 Hz vs our 5 Hz);
    # They say ["We predict the future 10 timestamps (4.0s) based on the historical 5 timestamps (2.0s)"]  
    # Hence, we match to our future window size (6.0s), but adjust the number of timesteps to account for
    # sampling rate differneces. 
    parser.add_argument('--future_length', type=int, default=30)
    parser.add_argument('--traj_scale', type=int, default=1)
    parser.add_argument('--learn_prior', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sample_k', type=int, default=5)
    # We save the model after every `num_epochs` epochs.  For each epoch we run `its_per_epoch` iterations.
    parser.add_argument('--decay_step', type=int, default=10)
    parser.add_argument('--decay_gamma', type=float, default=0.5)
    parser.add_argument('--iternum_print', type=int, default=100)

    parser.add_argument('--ztype', default='gaussian')
    parser.add_argument('--zdim', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=4)
    parser.add_argument('--p2p', action='store_true')
    parser.add_argument('--no_p2p', dest='p2p', action='store_false')
    parser.set_defaults(p2p=True)
    parser.add_argument('--hyper_scales', nargs='+', type=int,default=[5,11])
    parser.add_argument('--num_decompose', type=int, default=2)
    parser.add_argument('--min_clip', type=float, default=2.0)
    parser.add_argument('--model_save_epoch', type=int, default=10)
    parser.add_argument('--epoch_continue', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    losses_by_epoch_then_iteration=train(args)
    first_step_reduces_loss = losses_by_epoch_then_iteration[0][1]<losses_by_epoch_then_iteration[0][0]
    assert first_step_reduces_loss