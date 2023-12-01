import os
import sys
import argparse
import numpy as np
import torch
import random
from torch import optim
from torch.optim import lr_scheduler
sys.path.append(os.getcwd())

from groupnet.model.GroupNet_nba import GroupNet
from groupnet.compare.data_loader import make_data_loader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--num_train_games", type=int) # in [1,5,20]
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', default='nba')
parser.add_argument('--batch_size', type=int, default=32)
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
parser.add_argument('--future_length', type=int, default=15)
parser.add_argument('--traj_scale', type=int, default=1)
parser.add_argument('--learn_prior', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--sample_k', type=int, default=20)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--decay_step', type=int, default=10)
parser.add_argument('--decay_gamma', type=float, default=0.5)
parser.add_argument('--iternum_print', type=int, default=100)

parser.add_argument('--ztype', default='gaussian')
parser.add_argument('--zdim', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--hyper_scales', nargs='+', type=int,default=[5,11])
parser.add_argument('--num_decompose', type=int, default=2)
parser.add_argument('--min_clip', type=float, default=2.0)

parser.add_argument('--model_save_epoch', type=int, default=1)

parser.add_argument('--epoch_continue', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

""" dir """
MODEL_SAVE_DIR = f'saved_models/nba/{args.num_train_games}_train_games/'


""" setup """
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available(): 
    torch.cuda.set_device(args.gpu)
print('device:',device)
print(args)

def train(train_loader,epoch):
    model.train()
    total_iter_num = len(train_loader)
    iter_num = 0
    for data in train_loader:
        total_loss,loss_pred,loss_recover,loss_kl,loss_diverse = model(data)
        """ optimize """
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if iter_num % args.iternum_print == 0:
            print('Epochs: {:02d}/{:02d}| It: {:04d}/{:04d} | Total loss: {:03f}| Loss_pred: {:03f}| Loss_recover: {:03f}| Loss_kl: {:03f}| Loss_diverse: {:03f}'
            .format(epoch,args.num_epochs,iter_num,total_iter_num,total_loss.item(),loss_pred,loss_recover,loss_kl,loss_diverse))
        iter_num += 1

    scheduler.step()
    model.step_annealer()


""" model & optimizer """
model = GroupNet(args,device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)

""" Loading if needed """
if args.epoch_continue > 0:
    checkpoint_path = os.path.join(MODEL_SAVE_DIR,str(args.epoch_continue)+'.p')
    print('load model from: {checkpoint_path}')
    model_load = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(model_load['model_dict'])
    if 'optimizer' in model_load:
        optimizer.load_state_dict(model_load['optimizer'])
    if 'scheduler' in model_load:
        scheduler.load_state_dict(model_load['scheduler'])

""" dataloader """
train_loader = make_data_loader(f"train_{args.num_train_games}", args.past_length, args.future_length, args.batch_size)

""" start training """
model.set_device(device)
for epoch in range(args.epoch_continue, args.num_epochs):
    train(train_loader,epoch)
    """ save model """
    if  (epoch + 1) % args.model_save_epoch == 0:
        model_saved = {'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch + 1,'model_cfg': args}
        saved_path = os.path.join(MODEL_SAVE_DIR,str(epoch+1)+'.p')
        torch.save(model_saved, saved_path)



