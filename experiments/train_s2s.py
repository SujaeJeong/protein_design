from __future__ import print_function
import json, time, os, sys, glob
import shutil

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

from Bio.PDB import *
from Bio.SVDSuperimposer import SVDSuperimposer

# Library code
sys.path.insert(0, '..')
from struct2seq import *
from peptidebuilder.Geometry import *
from peptidebuilder.PeptideBuilder import *
from utils import *

args, device, model = setup_cli_model()
seq_optimizer = noam_opt.get_std_opt(model.seq_model.parameters(), args)
str_optimizer = noam_opt.get_std_opt(model.str_model.parameters(), args)
criterion = torch.nn.NLLLoss(reduction='none')

# Load the dataset
dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)

# Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
with open(args.file_splits) as f:
    dataset_splits = json.load(f)
train_set, validation_set, test_set = [
    Subset(dataset, [
        dataset_indices[chain_name] for chain_name in dataset_splits[key] 
        if chain_name in dataset_indices
    ])
    for key in ['train', 'validation', 'test']
]
loader_train, loader_validation, loader_test = [data.StructureLoader(
    d, batch_size=args.batch_tokens
) for d in [train_set, validation_set, test_set]]
print('Training:{}, Validation:{}, Test:{}'.format(len(train_set),len(validation_set),len(test_set)))

# Build basepath for experiment
if args.name != '':
    base_folder = 'log/' + args.name + '/'
else:
    base_folder = time.strftime('log/%y%b%d_%I%M%p/', time.localtime())

if os.path.exists(base_folder):
    raise Exception('Experiment already Done')

if not os.path.exists(base_folder):
    os.makedirs(base_folder)
subfolders = ['checkpoints', 'plots', 'structs', 'codes']
for subfolder in subfolders:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)
        
codelist = ['train_s2s.py', 'utils.py', '../struct2seq/data.py', '../struct2seq/struct2seq.py',
           '../struct2seq/self_attention.py', '../struct2seq/protein_features.py', '../struct2seq/noam_opt.py']
for codefile in codelist:
    newname = codefile.split('/')[-1]
    shutil.copyfile(codefile, base_folder + 'codes/' + newname)
    
# Log files
logfile = base_folder + 'log.txt'
with open(logfile, 'w') as f:
    f.write('Epoch\tTrain\tValidation\n')
with open(base_folder + 'args.json', 'w') as f:
    json.dump(vars(args), f)

start_train = time.time()
epoch_losses_train, epoch_losses_valid = [], []
epoch_checkpoints = []
total_step = 0
prev_elapsed = 0.
mse_loss = nn.MSELoss(reduction = 'sum')
io = PDBIO()

for e in range(args.epochs):
    # Training epoch
    model.train()
    train_sum, train_weights = 0., 0.
    for train_i, batch in enumerate(loader_train):
        seq_optimizer.zero_grad()
        str_optimizer.zero_grad()
        start_batch = time.time()        

        # Get a batch
        X, S, mask, lengths, gapmask, cath = featurize(batch, device, shuffle_fraction=args.shuffle)
        recon_V, log_probs, mu, logvar = model(X, S, lengths, mask)
        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask*gapmask.float(), weight=args.smoothing) 
        
        rmsds = 0.
        struct_count = 0
        phis, psis, omegas = np.split(recon_V.detach().cpu().numpy() / np.pi * 180., 3, axis = -1)
        
        for i in range(len(batch)):
            temp_seq = 'G' * int(mask[i].sum())
            end_i = lengths[i]
            recon_structure = make_structure(temp_seq, phis[i, :end_i, 0], psis[i, :end_i, 0], omegas[i, :end_i, 0])      
            rmsd = cal_rmsd(recon_structure, X[i], mask[i], gapmask[i])
            if rmsd > 0:           
                rmsds += rmsd
                struct_count+=1
        rmsds = rmsds / struct_count
        
        cos_dihedral = torch.cos(recon_V)[gapmask]
        sin_dihedral = torch.sin(recon_V)[gapmask]
        
        recon_loss = mse_loss(model.features._dihedrals(X)[gapmask], torch.cat((cos_dihedral, sin_dihedral), -1)) / torch.sum(gapmask)
        kl_div_loss = -0.5 * torch.sum(1 + logvar-mu.pow(2) - logvar.exp()) / torch.sum(gapmask)
        vae_loss = recon_loss + kl_div_loss * args.kl_div
        total_loss = rmsds + loss_av_smoothed + vae_loss
        total_loss.backward()
        str_optimizer.step()
        seq_optimizer.step()

        loss, loss_av = loss_nll(S, log_probs, gapmask.float())

        # Timing
        elapsed_train = time.time() - start_train
        time_elapsed = elapsed_train - prev_elapsed
        prev_elapsed = elapsed_train
        total_step += 1
        
        line = 'EPOCH{:>3}  STEP{:>7}  TOOK{:6.2f}  TOTAL_LOSS{:10.2f}  RMSD{:7.2f}  LAST_RMSD{:7.2f}  KL_DIV{:10.2f}  RECON_LOSS{:10.2f}  Cross Entropy{:7.2f}'.format(e, total_step, time_elapsed, total_loss, rmsds, rmsd, kl_div_loss, recon_loss, np.exp(loss_av.cpu().data.numpy()))
        print(line)

        if total_step % 50 == 0:
            io.set_structure(recon_structure)
            io.save(base_folder + f'structs/recon_{total_step}.pdb')
            os.system('cp ../data/cath/mmtf/{}.mmtf.gz {}structs/raw_{}.mmtf.gz'.format(batch[-1]['name'][:4], base_folder, total_step))
            plot_angle_distribution(phis, psis, omegas, S, total_step, mask, folder='{}plots/train_'.format(base_folder))

            with open(logfile, 'a') as f:
                f.write(line + '\n')
                
        train_sum += torch.sum(loss * gapmask.float()).cpu().data.numpy()
        train_weights += torch.sum(gapmask.float()).cpu().data.numpy()

        if total_step % 5000 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': str_optimizer.optimizer.state_dict(),
                'optimizer_state_dict': seq_optimizer.optimizer.state_dict()
            }, base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step))

    # Train image
    plot_log_probs(log_probs, total_step, folder='{}plots/train_'.format(base_folder))
    plot_angle_distribution(phis, psis, omegas, S, total_step, mask, folder='{}plots/train_'.format(base_folder))
    

    # Validation epoch
    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights = 0., 0.
        for _, batch in enumerate(loader_validation):
            X, S, mask, lengths, gapmask, cath = featurize(batch, device, shuffle_fraction=args.shuffle)
            recon_V, log_probs, mu, logvar = model(X, S, lengths, mask)
            loss, loss_av = loss_nll(S, log_probs, gapmask.float())
            
            rmsds = 0.
            phis, psis, omegas = np.split(recon_V.detach().cpu().numpy() / np.pi * 180., 3, axis = -1)
            struct_count = 0


            for i in range(len(batch)):
                temp_seq = 'G' * int(mask[i].sum())
                end_i = lengths[i]
                recon_structure = make_structure(temp_seq, phis[i, :end_i, 0], psis[i, :end_i, 0], omegas[i, :end_i, 0])      
                rmsd = cal_rmsd(recon_structure, X[i], mask[i], gapmask[i])
                if rmsd > 0:           
                    rmsds += rmsd
                    struct_count+=1
            rmsds = rmsds / struct_count
            
            cos_dihedral = torch.cos(recon_V)[gapmask]
            sin_dihedral = torch.sin(recon_V)[gapmask]

            recon_loss = mse_loss(model.features._dihedrals(X)[gapmask], torch.cat((cos_dihedral, sin_dihedral), -1)) / torch.sum(gapmask)
            kl_div_loss = -0.5 * torch.sum(1 + logvar-mu.pow(2) - logvar.exp()) / torch.sum(gapmask)
            vae_loss = recon_loss + kl_div_loss * args.kl_div
            total_loss = rmsds + loss_av_smoothed + vae_loss
            
            # Accumulate
            validation_sum += torch.sum(loss * gapmask.float()).cpu().data.numpy() 
            validation_weights += torch.sum(gapmask.float()).cpu().data.numpy()
            
            line = 'VALID  EPOCH{:>3}  STEP{:>7}  TOTAL_LOSS{:10.2f}  RMSD{:7.2f}  LAST_RMSD{:7.2f}  KL_DIV{:10.2f}  RECON_LOSS {:10.2f}  CROSS_ENTROPY{:10.2f}'.format(e, total_step, total_loss, rmsds, rmsd, kl_div_loss, recon_loss, np.exp(loss_av.cpu().data.numpy()))
            
            with open(logfile, 'a') as f:
                f.write(line + '\n')   
                
        io.set_structure(recon_structure)
        io.save(base_folder + f'structs/val_{e}.pdb')
        os.system('cp ../data/cath/mmtf/{}.mmtf.gz {}structs/valid_raw_{}.mmtf.gz'.format(batch[-1]['name'][:4], base_folder, e))
            

    train_loss = train_sum / train_weights
    train_perplexity = np.exp(train_loss)
    validation_loss = validation_sum / validation_weights
    validation_perplexity = np.exp(validation_loss)
    print('Perplexity\tTrain:{}\t\tValidation:{}'.format(train_perplexity, validation_perplexity))

    # Validation image
    plot_log_probs(log_probs, total_step, folder='{}plots/valid_{}_'.format(base_folder, batch[-1]['name']))
    plot_angle_distribution(phis, psis, omegas, S, total_step, mask, folder='{}plots/valid_'.format(base_folder))

    with open(logfile, 'a') as f:
        f.write('VALID  {}  {:8.4f}  {:8.4f}\n'.format(e, train_perplexity, validation_perplexity))

    # Save the model
    checkpoint_filename = base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step)
    torch.save({
        'epoch': e,
        'hyperparams': vars(args),
        'model_state_dict': model.state_dict(),
        'str_optimizer_state_dict': str_optimizer.optimizer.state_dict(),
        'seq_optimizer_state_dict': seq_optimizer.optimizer.state_dict()
    }, checkpoint_filename)

    epoch_losses_valid.append(validation_perplexity)
    epoch_losses_train.append(train_perplexity)
    epoch_checkpoints.append(checkpoint_filename)

# Determine best model via early stopping on validation
best_model_idx = np.argmin(epoch_losses_valid).item()
best_checkpoint = epoch_checkpoints[best_model_idx]
train_perplexity = epoch_losses_train[best_model_idx]
validation_perplexity = epoch_losses_valid[best_model_idx]
best_checkpoint_copy = base_folder + 'best_checkpoint_epoch{}.pt'.format(best_model_idx + 1)
shutil.copy(best_checkpoint, best_checkpoint_copy)
load_checkpoint(best_checkpoint_copy, model)


# Test epoch
model.eval()
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    for _, batch in enumerate(loader_test):
        X, S, mask, lengths, gapmask, cath = featurize(batch, device, shuffle_fraction=args.shuffle)
        recon_V, log_probs, mu, logvar = model(X, S, lengths, mask)
        loss, loss_av = loss_nll(S, log_probs, gapmask.float())
        # Accumulate
        
        rmsds = 0.
        struct_count = 0
        phis, psis, omegas = np.split(recon_V.detach().cpu().numpy() / np.pi * 180., 3, axis = -1)

        for i in range(len(batch)):
            temp_seq = 'G' * int(mask[i].sum())
            end_i = lengths[i]
            recon_structure = make_structure(temp_seq, phis[i, :end_i, 0], psis[i, :end_i, 0], omegas[i, :end_i, 0])      
            rmsd = cal_rmsd(recon_structure, X[i], mask[i], gapmask[i])
            if rmsd > 0:           
                rmsds += rmsd
                struct_count+=1
        rmsds = rmsds / struct_count
        
        cos_dihedral = torch.cos(recon_V)[gapmask]
        sin_dihedral = torch.sin(recon_V)[gapmask]
        
        recon_loss = mse_loss(model.features._dihedrals(X)[:, 1:-1][gapmask], torch.cat((cos_dihedral, sin_dihedral), -1)[:, 1:-1]) / torch.sum(gapmask)
        kl_div_loss = -0.5 * torch.sum(1 + logvar-mu.pow(2) - logvar.exp()) / torch.sum(gapmask)
        vae_loss = recon_loss + kl_div_loss * args.kl_div
        total_loss = rmsds + loss_av_smoothed + vae_loss
        
        print('total_loss: ', total_loss)        
        
        test_sum += torch.sum(loss * gapmask.float()).cpu().data.numpy()
        test_weights += torch.sum(gapmask.float()).cpu().data.numpy()
        
        

test_loss = test_sum / test_weights
test_perplexity = np.exp(test_loss)
print('Perplexity\tTest:{}'.format(test_perplexity))

with open(base_folder + 'results.txt', 'w') as f:
    f.write('Best epoch: {}\nPerplexities:\n\tTrain: {}\n\tValidation: {}\n\tTest: {}'.format(
        best_model_idx+1, train_perplexity, validation_perplexity, test_perplexity
    ))
