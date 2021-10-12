import argparse
import sys
import os
import data_utils
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import torch
from torch import nn
from model import RawGAT_ST  # In main model script we used our best RawGAT-ST-mul model. To use other models you need to call revelant model scripts from RawGAT_models folder
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def evaluate_accuracy(data_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()

    
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y, batch_meta in data_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        batch_out = model(batch_x,Freq_aug=False)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    sys_id_list = []
    
    score_list = []

    for batch_x, batch_y, batch_meta in data_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x,Freq_aug=False)
        
        batch_score = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel()     
        

        # add outputs
        fname_list.extend(list(batch_meta[1]))
        key_list.extend(
          ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        sys_id_list.extend([dataset.sysid_dict_inv[s.item()]
                            for s in list(batch_meta[3])])
        score_list.extend(batch_score.tolist())
        
    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            if dataset.is_eval:
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
            else:
                fh.write('{} {}\n'.format(f, cm))
    print('Result saved to {}'.format(save_path))

def train_epoch(data_loader, model, lr,optimizer, device):
    running_loss = 0
    num_total = 0.0
    model.train()

    # set objective (Loss) functions --> WCE
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y, batch_meta in data_loader:
        
        batch_size = batch_x.size(0)

        num_total += batch_size
        
        batch_x = batch_x.to(device)
       
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        
        batch_out = model(batch_x,Freq_aug=True)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss




if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019 RawGAT-ST model')
    
    # Dataset
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training, development and evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and  LA eval data folders are in the same database_path directory.')
    '''
    % database_path (full LA directory address)/
    %      |- ASVspoof2019_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default='/your/path/to/protocols/ASVspoof_database/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %      |- ASVspoof2019.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt 
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE',help='Weighted Cross Entropy Loss ')

    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='logical',choices=['logical', 'physical'], help='logical/physical')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--features', type=str, default='Raw_GAT')

    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 
    

    dir_yaml = os.path.splitext('model_config_RawGAT_ST')[0] + '.yaml'

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.load(f_yaml)
    
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    
    track = args.track
    assert track in ['logical', 'physical'], 'Invalid track given'
    is_logical = (track == 'logical')

    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    
    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    
    transforms = transforms.Compose([
        lambda x: pad(x),
        lambda x: Tensor(x)
        
    ])


    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    # validation Dataloader
    dev_set = data_utils.ASVDataset(database_path=args.database_path,protocols_path=args.protocols_path,is_train=False, is_logical=is_logical,
                                    transform=transforms,
                                    feature_name=args.features, is_eval=args.is_eval, eval_part=args.eval_part)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)
    
    
    #model 
    model = RawGAT_ST(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =(model).to(device)

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))


    # Inference
    if args.eval:
        assert args.eval_output is not None, 'You must provide an output path'
        assert args.model_path is not None, 'You must provide model checkpoint'
        produce_evaluation_file(dev_set, model, device, args.eval_output)
        sys.exit(0)

    # Training Dataloader
    train_set = data_utils.ASVDataset(database_path=args.database_path,protocols_path=args.protocols_path,is_train=True, is_logical=is_logical, transform=transforms,
                                      feature_name=args.features)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(num_epochs):
        
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device)
        val_loss = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch,
                                                   running_loss,val_loss))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
