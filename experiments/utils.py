from __future__ import print_function
import json, time, os, sys
import argparse
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Bio.PDB import *
from Bio.SVDSuperimposer import SVDSuperimposer
# Library code
sys.path.insert(0, '..')
from struct2seq import *
from argparse import ArgumentParser


def get_config():
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")
    
    ## Model Parameters
    # Model
    args.hidden = 256
    args.z_dim = 30
    args.layer = 'MPNNLayer'    
    args.mpnn = False
    args.bidirectional = False    
    args.num_encoder_layers = 2
    args.num_decoder_layers = 2
    
    # Input
    args.vocab_size = 20    
    args.k_neighbors = 30
    args.features = 'full'
    args.model_type = 'structure'    
    
    # Optimize
    args.dropout = 0.1
    args.smoothing = 0.1
    args.warmup = 10000
    args.kl_div = 1.        


    ## Running Parameters
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.name = 'MPNN1'    
    args.batch_tokens = 2500
    args.restore = ''
    args.file_data = '../data/domain_set.jsonl'
    args.file_splits = '../data/domain_set_splits.json'
    args.augment = ''
    args.epochs = 60
    args.shuffle = 0.
    args.seed = 1111

    return args

def setup_device_rng(args):
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # CUDA device handling.
    if torch.cuda.is_available():
        if not args.device == 'cuda':
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    return device

def setup_model(hyperparams, device, args):
    # Build the model
    if hyperparams['model_type'] == 'structure':
        model = struct2seq.Struct2Seq(
            num_letters=hyperparams['vocab_size'], 
            node_features=hyperparams['hidden'],
            edge_features=hyperparams['hidden'], 
            hidden_dim=hyperparams['hidden'],
            k_neighbors=hyperparams['k_neighbors'],
            protein_features=hyperparams['features'],
            dropout=hyperparams['dropout'],
            use_mpnn=hyperparams['mpnn'],
            args = args
        ).to(device)
    elif hyperparams['model_type'] == 'sequence':
        model = seq_model.SequenceModel(
            num_letters=hyperparams['vocab_size'],
            hidden_dim=hyperparams['hidden'],
            top_k=hyperparams['k_neighbors']
        ).to(device)
    elif hyperparams['model_type'] == 'rnn':
        model = seq_model.LanguageRNN(
            num_letters=hyperparams['vocab_size'],
            hidden_dim=hyperparams['hidden']
        ).to(device)

    print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    return model

def setup_cli_model():
    args = get_config()
    device = setup_device_rng(args)
    model = setup_model(vars(args), device, args)
    if args.restore is not '':
        load_checkpoint(args.restore, model)
    return args, device, model

def load_checkpoint(checkpoint_path, model):
    print('Loading checkpoint from {}'.format(checkpoint_path))
    state_dicts = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dicts['model_state_dict'])
    print('\tEpoch {}'.format(state_dicts['epoch']))
    return

def cal_rmsd(recon_c, raw_c, mask, gapmask):
    ### mask: max_len때문에 맞춰주는 것
    ### gapmask: gap때문에 맞춰주는 것
    try:
        mask = mask.bool().cpu().numpy()
        gapmask = gapmask.bool().cpu().numpy()


        raw_c = raw_c.view(-1, 3).cpu().numpy()
        target_atoms = ['N', 'CA', 'C', 'O']
        recon_coords = {c: list() for c in target_atoms}


        for atom in recon_c.get_atoms():
            atom_n = atom.get_name()
            if atom_n in target_atoms:
                recon_coords[atom_n].append(atom.get_coord())

        for c in target_atoms:
            recon_coords[c] = np.stack(recon_coords[c])
        recon_backbone = np.stack((recon_coords[c] for c in target_atoms), axis = 1).reshape(-1, 3)
        a = np.repeat(mask, 4)
        sup = SVDSuperimposer()    
        sup.set(raw_c[np.repeat(gapmask, 4)].reshape(-1, 3),
                recon_backbone[np.repeat(gapmask[mask], 4)].reshape(-1, 3))
        sup.run()
        rot, trans = sup.get_rotran()

        transform_c = np.dot(recon_backbone, rot) + trans
        diff = raw_c[np.repeat(gapmask, 4)] - transform_c[np.repeat(gapmask[mask], 4)]
        rmsd = np.sqrt(np.sum(diff * diff) / np.sum(gapmask* 4))
    except: 
        rmsd = -1
        
    return rmsd


def featurize(batch, device, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    gapmask = np.zeros([B, L_max], dtype=np.bool)
    caths = []
    

    def shuffle_subset(n, p):
        n_shuffle = np.random.binomial(n, p)
        ix = np.arange(n)
        ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
        ix_subset_shuffled = np.copy(ix_subset)
        np.random.shuffle(ix_subset_shuffled)
        ix[ix_subset] = ix_subset_shuffled
        return ix

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
        else:
            S[i, :l] = indices
        gapmask[i, :l] = b['gap_m'][:l]
        caths.append(b['CATH'])

    # Mask
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    gapmask = torch.from_numpy(gapmask).to(dtype=torch.bool, device=device)
    
    return X, S, mask, lengths, gapmask, caths

def plot_log_probs(log_probs, total_step, folder=''):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    reorder = 'DEKRHQNSTPGAVILMCFWY'
    permute_ix = np.array([alphabet.index(c) for c in reorder])
    plt.close()
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(111)
    P = np.exp(log_probs.cpu().data.numpy())[0].T
    plt.imshow(P[permute_ix])
    plt.clim(0,1)
    plt.colorbar()
    plt.yticks(np.arange(20), [a for a in reorder])
    ax.tick_params(
        axis=u'both', which=u'both',length=0, labelsize=5
    )
    plt.tight_layout()
    plt.savefig(folder + 'probs{}.pdf'.format(total_step))
    return

def plot_angle_distribution(phis, psis, omegas, S, total_step, mask, folder = 'plots/'):
    
    fig, axs = plt.subplots(2, 1, figsize = (6, 14))

    nonzero_idx = (phis!=0) & (psis != 0) & (omegas !=0) & mask.unsqueeze(2).cpu().numpy().astype(bool)
    aas = 'ACDEFGHIKLMNPQRSTVWY'
    aas = ','.join(aas).split(',')
    seq_bins = np.arange(21)
    omega_bins = np.arange(38) * 10 - 185 ### -185 ~ 185

    ## ramachandran plot
    axs[0].scatter(phis[nonzero_idx], psis[nonzero_idx], s = 2)
    axs[0].set_title('Ramchandran plot', size = 15)
    axs[0].set_xlabel('PHI angle (degree)', size = 12)
    axs[0].set_ylabel('PSI angle (degree)', size = 12)
    axs[0].set_aspect('equal', 'box')
    axs[0].set_xlim(-180, 180)
    axs[0].set_ylim(-180, 180)
    axs[0].set_xticks(np.arange(7) * 60 - 180)
    axs[0].set_yticks(np.arange(7) * 60 - 180)
    axs[0].grid(True)

    # omega Vs residue
    seq_proc = S.unsqueeze(2).cpu().numpy()[nonzero_idx]
    omega_proc = ((omegas[nonzero_idx] + 270+180)%360)-180  ### shift 90'
    omega_counts = np.histogram2d(seq_proc, omega_proc, bins = [seq_bins, omega_bins])[0]
    omega_ratio = omega_counts / (np.maximum(omega_counts.sum(axis = 1, keepdims = True), 1))
    im = axs[1].imshow(omega_ratio.T, cmap = 'Reds')
    axs[1].set_xticks(np.arange(21))
    axs[1].set_yticks(9 * np.arange(5))
    axs[1].set_xticklabels(aas)
    axs[1].set_yticklabels(['-90', '0', '90', '180', '270'])
    axs[1].set_xlabel('AA Residue', size = 12)
    axs[1].set_ylabel('OMEGA angle (degree)', size = 12)
    axs[1].set_title('OMEGA angle per residue', size = 15)
    axs[1].set_xlim(-0.5, 19.5)
    axs[1].set_ylim(-0.5, 36.5)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = axs[1].figure.colorbar(im, cax = cax)
    im.set_clim(0, 1)
    plt.tight_layout()
    plt.savefig(folder + 'angle_plots{}.pdf'.format(total_step))
    plt.close()

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def check_gap(mask):
    if torch.abs(mask[1:]-mask[:-1]).sum(dim = -1) > 1:
        if mask[0] + mask[-1] == 0:
            return False
        else:
            return True
    else:
        return False

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed_reweight(S, log_probs, mask, weight=0.1, factor=10.):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    # Upweight the examples with worse performance
    loss = -(S_onehot * log_probs).sum(-1)
    
    # Compute an error-weighted average
    loss_av_per_example = torch.sum(loss * mask, -1, keepdim=True) / torch.sum(mask, -1, keepdim=True)
    reweights = torch.nn.functional.softmax(factor * loss_av_per_example, 0)
    mask_reweight = mask * reweights
    loss_av = torch.sum(loss * mask_reweight) / torch.sum(mask_reweight)
    return loss, loss_av
