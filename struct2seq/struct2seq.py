from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attention import *
from .protein_features import ProteinFeatures

class Struct2Seq(nn.Module):
    def __init__(self, num_letters, node_features, edge_features, 
        hidden_dim, args, 
        vocab=20, k_neighbors=30, protein_features='full', augment_eps=0.,
        dropout=0.1, forward_attention_decoder=True, use_mpnn=False):
        """ Graph labeling network """
        super(Struct2Seq, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        num_encoder_layers = args.num_encoder_layers
        num_decoder_layers = args.num_decoder_layers

        # Featurization layers
        self.features = ProteinFeatures(
            node_features, edge_features, args = args, top_k=k_neighbors,
            features_type=protein_features, augment_eps=augment_eps,
            dropout=dropout
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        if args.layer == 'TransformerLayer':
            layer = TransformaerLayer
        elif args.layer == 'MPNNLayer':
            layer = MPNNLayer
        elif args.layer == 'LSTMLayer':
            layer = LSTMLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim*2, args, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.fc_mu = nn.Linear(hidden_dim, args.z_dim)
        self.fc_sig = nn.Linear(hidden_dim, args.z_dim)
        
        self.generator = LSTMLayer2(hidden_dim, hidden_dim*2, args.z_dim, dropout=dropout)
        self.recon_out = nn.Linear(hidden_dim, 3)
        
        self.fc_decode = nn.Linear(args.z_dim, args.hidden)
        # Decoder layers
        self.forward_attention_decoder = forward_attention_decoder
        self.decoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim*3, args, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        self.str_model = nn.ModuleList([self.features, self.W_v, self.W_e, self.encoder_layers, 
                                       self.fc_mu, self.fc_sig, self.generator, self.recon_out])
        self.seq_model = nn.ModuleList([self.features, self.W_v, self.W_e, self.encoder_layers, self.fc_mu, self.fc_sig, self.fc_decode, self.decoder_layers, self.W_out])

    def _autoregressive_mask(self, E_idx):
        N_nodes = E_idx.size(1) # B N K
        ii = torch.arange(N_nodes) # N
        ii = ii.view((1, -1, 1)) # 1 N 1
        mask = E_idx - ii < 0 # B N K
        mask = mask.type(torch.float32)

        return mask

    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std 
    
    
    def forward(self, X, S, L, mask):
        """ Graph-conditioned sequence model """

        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)
            ### h_EV: [batch, n_nodes, hidden_dim*2]
            
        ###################    
        #    Generator    #
        ###################
        
        mu = self.fc_mu(h_V[:, -1, :])
        sig = self.fc_sig(h_V[:, -1, :])
        sample = self.sample_z(mu, sig)
        
        point = self.recon_out(self.generator(sample, h_EV, mask_V=mask))
        cos = torch.cos(point)
        sin = torch.sin(point)
        recon_V = torch.atan2(sin, cos)
        
        n_nodes = h_EV.shape[1]
        h_V = self.fc_decode(mu.unsqueeze(1).expand(-1, n_nodes,-1))
        
        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, E_idx)

        # Decoder uses masked self-attention
        mask_attend = self._autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend    #[B,N,1,1] * [B N K]
        
        if self.forward_attention_decoder:
            mask_fw = mask_1D * (1. - mask_attend)
            h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        else:
            h_ESV_encoder_fw = 0
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = layer(h_V, h_ESV, mask_V=mask)

        logits = self.W_out(h_V) 
        log_probs = F.log_softmax(logits, dim=-1)
        
        return recon_V, log_probs, mu, sig


    def sample(self, X, L, mask=None, temperature=1.0):
        """ Autoregressive decoding of a model """
         # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)
        
        # Decoder alternates masked self-attention
        mask_attend = self._autoregressive_mask(E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 20))
        h_S = torch.zeros_like(h_V)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder_layers))]
        for t in range(N_nodes):
            # Hidden layers
            E_idx_t = E_idx[:,t:t+1,:]
            h_E_t = h_E[:,t:t+1,:,:]
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            # Stale relational features for future states
            h_ESV_encoder_t = mask_fw[:,t:t+1,:,:] * cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)
            for l, layer in enumerate(self.decoder_layers):
                # Updated relational features for future states
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = h_V_stack[l][:,t:t+1,:]
                h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_ESV_encoder_t
                h_V_stack[l+1][:,t,:] = layer(
                    h_V_t, h_ESV_t, mask_V=mask[:,t:t+1]
                ).squeeze(1)

            # Sampling step
            h_V_t = h_V_stack[-1][:,t,:]
            logits = self.W_out(h_V_t) / temperature
            probs = F.softmax(logits, dim=-1)
            S_t = torch.multinomial(probs, 1).squeeze(-1)

            # Update
            h_S[:,t,:] = self.W_s(S_t)
            S[:,t] = S_t
        return S
