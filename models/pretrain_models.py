from vit_pytorch.vit import pair, Transformer
import math
import torch
from torch import nn
from einops.layers.torch import Rearrange

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from positional_encodings.torch_encodings import PositionalEncoding2D

from typing import Any, Dict, List, Optional, Tuple, Type, Union

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _GymFallback:
        class Space:
            pass

    class _SpacesFallback:
        class Space:
            pass

        class Box:
            pass

    gym = _GymFallback()
    spaces = _SpacesFallback()

try:
    from stable_baselines3.common.torch_layers import (
        BaseFeaturesExtractor,
        FlattenExtractor,
    )
    from stable_baselines3.common.type_aliases import Schedule
    from stable_baselines3.common.policies import ActorCriticPolicy
except ImportError:
    class BaseFeaturesExtractor(nn.Module):
        def __init__(self, observation_space=None, features_dim=0):
            super().__init__()
            self.observation_space = observation_space
            self.features_dim = features_dim

    class FlattenExtractor(nn.Flatten):
        pass

    Schedule = Any

    class ActorCriticPolicy(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ImportError("stable_baselines3 is required for MAEPolicy, but DPPO does not use it.")

import random

from vit_pytorch.vit import Transformer

try:
    from utils.pretrain_utils import vt_load
except ImportError:
    def vt_load(*args, **kwargs):
        raise ImportError("utils.pretrain_utils.vt_load requires optional SB3 logging dependencies.")

from tqdm import tqdm

import torch.optim as optim

import numpy as np

class EarlyCNN(nn.Module):
    def __init__(self, in_channels, encoder_dim, key='image'):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, encoder_dim//8, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(encoder_dim//8, encoder_dim//4, 4, stride=2, padding=1)
        if key == 'image':
            self.conv3 = nn.Conv2d(encoder_dim//4, encoder_dim//2, 4, stride=2, padding=1)
        else:
            self.conv3 = nn.Conv2d(encoder_dim//4, encoder_dim//2, 3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(encoder_dim//2, encoder_dim, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return self.conv4(x).flatten(2).transpose(1, 2)
    

class VTMAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        num_tactiles = 2,
        early_conv_masking = False,
        use_sincosmod_encodings = True,
        frame_stack = 1,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        self.num_tactiles = num_tactiles

        self.frame_stack = frame_stack

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        num_decoder_patches = num_patches - 1

        self.use_sincosmod_encodings = use_sincosmod_encodings

        print("num_patches: ", num_patches)
        print("num_decoder_patches: ", num_decoder_patches)

        self.early_conv_masking = early_conv_masking
        if self.early_conv_masking:
            self.early_conv_vision = EarlyCNN(self.encoder.image_channels, encoder_dim, key='image')
            self.early_conv_tactile = EarlyCNN(self.encoder.tactile_channels, encoder_dim, key='tactile')

        self.image_to_patch = encoder.image_to_patch_embedding[0]
        self.image_patch_to_emb = nn.Sequential(*encoder.image_to_patch_embedding[1:])
        pixel_values_per_patch = encoder.image_to_patch_embedding[2].weight.shape[-1]

        self.tactile_to_patch = encoder.tactile_to_patch_embedding[0]
        self.tactile_patch_to_emb = nn.Sequential(*encoder.tactile_to_patch_embedding[1:])
        tactile_values_per_patch = encoder.tactile_to_patch_embedding[2].weight.shape[-1]

        self.encoder_dim = encoder_dim

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_decoder_patches, decoder_dim) 
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.to_tactiles = nn.Linear(decoder_dim, tactile_values_per_patch)

        self.num_tactiles = num_tactiles

        enc_pos_embedding = PositionalEncoding2D(encoder_dim)

        sample_image = torch.zeros((1, self.encoder.image_height//self.encoder.image_patch_height, self.encoder.image_width//self.encoder.image_patch_width, encoder_dim))

        image_pos_embedding = enc_pos_embedding(sample_image).flatten(1,2) # 1 x 1image_patches x encoder_dim
        print("image_pos_embedding.shape: ", image_pos_embedding.shape)
        self.register_buffer('image_enc_pos_embedding', image_pos_embedding) # 1 x image_patches x encoder_dim
        
        sample_tactile = torch.zeros((1, self.encoder.tactile_height//self.encoder.tactile_patch_height, self.encoder.tactile_width//self.encoder.tactile_patch_width, encoder_dim))

        tactile_pos_embedding = enc_pos_embedding(sample_tactile).flatten(1,2) # 1 x 1tactile_patches x encoder_dim

        self.register_buffer('tactile_enc_pos_embedding', repeat(tactile_pos_embedding, 'b n d -> b (v n) d', v = self.num_tactiles)) # 1 x tactile_patches x encoder_dim

        sample_image = torch.zeros((1, self.encoder.image_height//self.encoder.image_patch_height, self.encoder.image_width//self.encoder.image_patch_width, decoder_dim))
        image_pos_embedding = enc_pos_embedding(sample_image).flatten(1,2) # 1 x 1image_patches x decoder_dim
        self.register_buffer('image_dec_pos_embedding', image_pos_embedding) # 1 x image_patches x decoder_dim

        sample_tactile = torch.zeros((1, self.encoder.tactile_height//self.encoder.tactile_patch_height, self.encoder.tactile_width//self.encoder.tactile_patch_width, decoder_dim))
        tactile_pos_embedding = enc_pos_embedding(sample_tactile).flatten(1,2) # 1 x 1tactile_patches x decoder_dim
        self.register_buffer('tactile_dec_pos_embedding', repeat(tactile_pos_embedding, 'b n d -> b (v n) d', v = self.num_tactiles)) # 1 x tactile_patches x decoder_dim

        self.encoder_modality_embedding = nn.Embedding((1 + self.num_tactiles), encoder_dim)
        self.decoder_modality_embedding = nn.Embedding((1 + self.num_tactiles), decoder_dim)
        

    def forward(self, x, use_vision=True, use_tactile=True):

        if 'image' in x.keys():
            device = x['image'].device
        else:
            device = x['tactile1'].device
            use_vision = False

        # get patches

        if use_vision:
            image_patches = self.image_to_patch(x['image']) 
            batch, num_image_patches, *_ = image_patches.shape
        else:
            image_patches = torch.zeros((x['tactile1'].shape[0], 0, 3)).to(device)
            num_image_patches = 0
        
        if self.num_tactiles > 0 and use_tactile:
            tactile_patches_list = []
            for i in range(1,self.num_tactiles+1):
                tactile_patches_list.append(self.tactile_to_patch(x['tactile'+str(i)]))

            tactile_patches = torch.cat(tactile_patches_list, dim=1) 
            batch, num_tactile_patches, *_ = tactile_patches.shape
        else:
            tactile_patches = torch.zeros((x['image'].shape[0], 0, 3)).to(device)
            num_tactile_patches = 0

        num_patches = num_image_patches + num_tactile_patches

        num_decoder_patches = num_patches

        # patch to encoder tokens and add positions

        if self.early_conv_masking:
            if use_vision:
                image_tokens = self.early_conv_vision(x['image'])
            else:
                image_tokens = torch.zeros((batch, 0, self.encoder_dim)).to(device)
            if self.num_tactiles > 0 and use_tactile:
                tactile_tokens_list = []
                for i in range(1,self.num_tactiles+1):
                    tactile_tokens_list.append(self.early_conv_tactile(x['tactile'+str(i)]))
                tactile_tokens = torch.cat(tactile_tokens_list, dim=1)
            else:
                tactile_tokens = torch.zeros((batch, 0, image_tokens.shape[-1])).to(device)
        else:
            if use_vision:
                image_tokens = self.image_patch_to_emb(image_patches) 
            else:
                image_tokens = torch.zeros((batch, 0, self.encoder_dim)).to(device)
            if self.num_tactiles > 0 and use_tactile:
                tactile_tokens = self.tactile_patch_to_emb(tactile_patches) 
            else:
                tactile_tokens = torch.zeros((batch, 0, image_tokens.shape[-1])).to(device)

        if use_vision:
            
            if self.use_sincosmod_encodings:
                image_tokens += self.encoder_modality_embedding(torch.tensor(0, device = device))
                image_tokens = image_tokens + self.image_enc_pos_embedding
        
        if self.num_tactiles > 0 and use_tactile:
            num_single_tactile_patches = num_tactile_patches//(self.num_tactiles)
            for i in range(self.num_tactiles):
                if self.use_sincosmod_encodings:
                    tactile_tokens[:, i*num_single_tactile_patches:(i+1)*num_single_tactile_patches] += self.encoder_modality_embedding(torch.tensor(1+i, device = device))
            if self.use_sincosmod_encodings:
                tactile_tokens = tactile_tokens + self.tactile_enc_pos_embedding
            
        tokens = torch.cat((image_tokens, tactile_tokens), dim=1)
        
        if not self.use_sincosmod_encodings:
           tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        image_perc = num_image_patches/num_patches
        num_masked_image = int(num_masked * image_perc)
        if self.num_tactiles > 0 and use_tactile:
            num_masked_tactile = (num_masked - num_masked_image)//self.num_tactiles

        rand_indices_image = torch.rand(batch, num_image_patches, device = device).argsort(dim = -1)
        masked_indices_image, unmasked_indices_image = rand_indices_image[:, :num_masked_image], rand_indices_image[:, num_masked_image:]

        if self.num_tactiles > 0 and use_tactile:
            masked_indices_tactile = []
            unmasked_indices_tactile = []
            count = num_image_patches
            for i in range(self.num_tactiles):
                rand_indices_tactile = torch.rand(batch, num_tactile_patches//self.num_tactiles, device = device).argsort(dim = -1)+count
                masked_indices_tactile.append(rand_indices_tactile[:, :num_masked_tactile])
                unmasked_indices_tactile.append(rand_indices_tactile[:, num_masked_tactile:])
                count += int(num_tactile_patches/self.num_tactiles)
            masked_indices_tactile = torch.cat(masked_indices_tactile, dim=1)
            unmasked_indices_tactile = torch.cat(unmasked_indices_tactile, dim=1)
        else:
            masked_indices_tactile = torch.zeros((batch, 0),dtype=torch.long).to(device)
            unmasked_indices_tactile = torch.zeros((batch, 0),dtype=torch.long).to(device)
    
        masked_indices = torch.cat((masked_indices_image, masked_indices_tactile), dim=1)
        unmasked_indices = torch.cat((unmasked_indices_image, unmasked_indices_tactile), dim=1)

        
        num_masked = masked_indices.shape[-1]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        
        # get the patches to be masked for the final reconstruction loss

        if not self.early_conv_masking: # This is a hack to deal with the fact that masked_indices_image have different lenghts per sample in the batch
            masked_image_patches = image_patches[batch_range, masked_indices_image]
            masked_tactile_patches = tactile_patches[batch_range, masked_indices_tactile-image_patches.shape[1]]
            
        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)
        
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        if self.use_sincosmod_encodings:
            unmasked_decoder_tokens = decoder_tokens
        else:
            unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        
        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        if not self.use_sincosmod_encodings:
            mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)    
        
        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_decoder_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        start_index = 0
        end_index = None
        
        decoder_image_tokens = decoder_tokens[:, start_index:num_image_patches+start_index]
        decoder_tactile_tokens = decoder_tokens[:, num_image_patches+start_index:end_index]

        if use_vision:
            if self.use_sincosmod_encodings:
                decoder_image_tokens += self.decoder_modality_embedding(torch.tensor(0, device=device))
                decoder_image_tokens = decoder_image_tokens + self.image_dec_pos_embedding
        
        if self.num_tactiles > 0 and use_tactile:
            num_single_tactile_patches = num_tactile_patches//(self.num_tactiles)
            for i in range(self.num_tactiles):
                if self.use_sincosmod_encodings:
                    decoder_tactile_tokens[:, i*num_single_tactile_patches:(i+1)*num_single_tactile_patches] += self.decoder_modality_embedding(torch.tensor(1+i, device=device))
            if self.use_sincosmod_encodings:
                decoder_tactile_tokens = decoder_tactile_tokens + self.tactile_dec_pos_embedding

        decoder_tokens[:,start_index:end_index] = torch.cat((decoder_image_tokens, decoder_tactile_tokens), dim=1)

        decoded_tokens = self.decoder(decoder_tokens)
        
        if self.early_conv_masking:
            image_tokens = decoded_tokens[:, :num_image_patches]
            tactile_tokens = decoded_tokens[:, num_image_patches:]
            
            pred_pixel_values = self.to_pixels(image_tokens)
            pred_tactile_values = self.to_tactiles(tactile_tokens)

            recon_loss = 0
            if self.num_tactiles > 0 and use_tactile:
                recon_loss += 10*F.mse_loss(pred_tactile_values, tactile_patches)
            if use_vision:
                recon_loss += F.mse_loss(pred_pixel_values, image_patches)

        else:
        # splice out the mask tokens and project to pixel values

            mask_image_tokens = decoded_tokens[batch_range, masked_indices_image]
            pred_pixel_values = self.to_pixels(mask_image_tokens)
            
            # splice out the mask tokens and project to tactile values

            mask_tactile_tokens = decoded_tokens[batch_range, masked_indices_tactile]
            pred_tactile_values = self.to_tactiles(mask_tactile_tokens)
                
            # calculate reconstruction loss
            recon_loss = 0
            if self.num_tactiles > 0 and use_tactile:
                recon_loss += 10*F.mse_loss(pred_tactile_values, masked_tactile_patches)
            if use_vision:
                recon_loss += F.mse_loss(pred_pixel_values, masked_image_patches)

        return recon_loss
    
    def reconstruct(self, x, mask_ratio=None, use_vision=True, use_tactile=True):
    
        if mask_ratio is None:
            mask_ratio = self.masking_ratio

        if 'image' in x.keys():
            device = x['image'].device
        else:
            device = x['tactile1'].device
            use_vision = False

        # get patches
        if use_vision:
            image_patches = self.image_to_patch(x['image'])
            batch, num_image_patches, *_ = image_patches.shape
        else:
            image_patches = torch.zeros((x['tactile1'].shape[0], 0, 3)).to(device)
            num_image_patches = 0
        
        if self.num_tactiles > 0 and use_tactile:
            tactile_patches_list = []
            for i in range(1,self.num_tactiles+1):
                tactile_patches_list.append(self.tactile_to_patch(x['tactile'+str(i)]))

            tactile_patches = torch.cat(tactile_patches_list, dim=1)
            batch, num_tactile_patches, *_ = tactile_patches.shape
        else:
            tactile_patches = torch.zeros((x['image'].shape[0], 0, 3)).to(device)
            num_tactile_patches = 0
        
        num_patches = num_image_patches + num_tactile_patches

        num_decoder_patches = num_patches


        # patch to encoder tokens and add positions

        if self.early_conv_masking:
            if use_vision:
                image_tokens = self.early_conv_vision(x['image'])
            else:
                image_tokens = torch.zeros((batch, 0, self.encoder_dim)).to(device)
            if self.num_tactiles > 0 and use_tactile:
                tactile_tokens_list = []
                for i in range(1,self.num_tactiles+1):
                    tactile_tokens_list.append(self.early_conv_tactile(x['tactile'+str(i)]))
                tactile_tokens = torch.cat(tactile_tokens_list, dim=1)
            else:
                tactile_tokens = torch.zeros((batch, 0, image_tokens.shape[-1])).to(device)
        else:
            if use_vision:
                image_tokens = self.image_patch_to_emb(image_patches)
            else:
                image_tokens = torch.zeros((batch, 0, self.encoder_dim)).to(device)
            if self.num_tactiles > 0 and use_tactile:
                tactile_tokens = self.tactile_patch_to_emb(tactile_patches)
            else:
                tactile_tokens = torch.zeros((batch, 0, image_tokens.shape[-1])).to(device)
        
        if use_vision:
            if self.use_sincosmod_encodings:
                image_tokens += self.encoder_modality_embedding(torch.tensor(0, device=device))
                image_tokens = image_tokens + self.image_enc_pos_embedding
            
        if self.num_tactiles > 0 and use_tactile:
            num_single_tactile_patches = num_tactile_patches//(self.num_tactiles)
            for i in range(self.num_tactiles):
                if self.use_sincosmod_encodings:
                    tactile_tokens[:, i*num_single_tactile_patches:(i+1)*num_single_tactile_patches] += self.encoder_modality_embedding(torch.tensor(1+i, device=device))
            if self.use_sincosmod_encodings:
                tactile_tokens = tactile_tokens + self.tactile_enc_pos_embedding

        tokens = torch.cat((image_tokens, tactile_tokens), dim=1)
        if not self.use_sincosmod_encodings:
            tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        image_patches_vis = image_patches.clone()
        tactile_patches_vis = tactile_patches.clone()
        
        if use_vision:
            num_masked_image = int(mask_ratio * num_image_patches)
            rand_indices_image = torch.rand(batch, num_image_patches, device = device).argsort(dim = -1)
            masked_indices_image, unmasked_indices_image = rand_indices_image[:, :num_masked_image], rand_indices_image[:, num_masked_image:]
        else:
            masked_indices_image = torch.zeros((batch, 0),dtype=torch.long).to(device)
            unmasked_indices_image = torch.zeros((batch, 0),dtype=torch.long).to(device)

        if self.num_tactiles > 0 and use_tactile:
            num_masked_tactile = int(mask_ratio * num_tactile_patches / self.num_tactiles)
            masked_indices_tactile = []
            unmasked_indices_tactile = []
            count = num_image_patches
            for i in range(self.num_tactiles):
                rand_indices_tactile = torch.rand(batch, num_tactile_patches//self.num_tactiles, device = device).argsort(dim = -1)+count
                masked_indices_tactile.append(rand_indices_tactile[:, :num_masked_tactile])
                unmasked_indices_tactile.append(rand_indices_tactile[:, num_masked_tactile:])
                count += int(num_tactile_patches/self.num_tactiles)
            masked_indices_tactile = torch.cat(masked_indices_tactile, dim=1)
            unmasked_indices_tactile = torch.cat(unmasked_indices_tactile, dim=1)
        else:
            masked_indices_tactile = torch.zeros((batch, 0),dtype=torch.long).to(device)
            unmasked_indices_tactile = torch.zeros((batch, 0),dtype=torch.long).to(device)
        
        masked_indices = torch.cat((masked_indices_image, masked_indices_tactile), dim=1)
        unmasked_indices = torch.cat((unmasked_indices_image, unmasked_indices_tactile), dim=1)

        num_masked = masked_indices.shape[-1]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        if not self.early_conv_masking: # Hack: see forward() for explanation
            masked_image_patches = image_patches[batch_range, masked_indices_image].clone()
            masked_tactile_patches = tactile_patches[batch_range, masked_indices_tactile-image_patches.shape[1]].clone()

        # set the masked patches to 0.5, for visualization (h: num_patches_in_height, w: num_patches_in_width, c: num_channels)
        if use_vision:
            image_transform = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                                h = self.encoder.image_height//self.encoder.image_patch_height, w = self.encoder.image_width//self.encoder.image_patch_width, 
                                p1 = self.encoder.image_patch_height, p2 = self.encoder.image_patch_width)
            if not self.early_conv_masking:
                image_patches[batch_range, masked_indices_image] = 0.5
                image_masked = image_transform(image_patches)
            else:
                image_patches_vis[batch_range, masked_indices_image] = 0.5
                image_masked = image_transform(image_patches_vis)
        
        if self.num_tactiles > 0 and use_tactile:
           
            tactile_transform = Rearrange('b (n h w) (p1 p2 c) -> b (n c) (h p1) (w p2)',
                                h = self.encoder.tactile_height//self.encoder.tactile_patch_height, w = self.encoder.tactile_width//self.encoder.tactile_patch_width,
                                p1 = self.encoder.tactile_patch_height, p2 = self.encoder.tactile_patch_width)
            if not self.early_conv_masking:
                tactile_patches[batch_range, masked_indices_tactile-image_patches.shape[1]] = np.inf
                tactile_masked = tactile_transform(tactile_patches)
            else:
                tactile_patches_vis[batch_range, masked_indices_tactile-image_patches.shape[1]] = np.inf
                tactile_masked = tactile_transform(tactile_patches_vis)
            
        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)
        
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        if self.use_sincosmod_encodings:
            unmasked_decoder_tokens = decoder_tokens 
        else:
            unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        if not self.use_sincosmod_encodings:
            mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_decoder_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        start_index = 0
        end_index = None
        
        decoder_image_tokens = decoder_tokens[:, start_index:num_image_patches+start_index]
        decoder_tactile_tokens = decoder_tokens[:, num_image_patches+start_index:end_index]

        if use_vision:
            if self.use_sincosmod_encodings:
                decoder_image_tokens += self.decoder_modality_embedding(torch.tensor(0, device = device))
                decoder_image_tokens = decoder_image_tokens + self.image_dec_pos_embedding
            
        
        if self.num_tactiles > 0 and use_tactile:
            num_single_tactile_patches = num_tactile_patches//self.num_tactiles
            for i in range(self.num_tactiles):
                if self.use_sincosmod_encodings:
                    decoder_tactile_tokens[:, i*num_single_tactile_patches:(i+1)*num_single_tactile_patches] += self.decoder_modality_embedding(torch.tensor(1+i, device = device))
            if self.use_sincosmod_encodings:
                decoder_tactile_tokens = decoder_tactile_tokens + self.tactile_dec_pos_embedding 


        decoder_tokens[:, start_index:end_index] = torch.cat((decoder_image_tokens, decoder_tactile_tokens), dim=1)

        decoded_tokens = self.decoder(decoder_tokens)

        if self.early_conv_masking:
            if use_vision:
                image_tokens = decoded_tokens[:, :num_image_patches]
                pred_pixel_values = self.to_pixels(image_tokens)
                
                recon_loss_image = F.mse_loss(pred_pixel_values, image_patches)
                image_patches = pred_pixel_values
                image_rec = image_transform(image_patches)

            if self.num_tactiles > 0 and use_tactile:
                tactile_tokens = decoded_tokens[:, num_image_patches:]
                pred_tactile_values = self.to_tactiles(tactile_tokens)
                recon_loss_tactile = F.mse_loss(pred_tactile_values, tactile_patches)
                tactile_patches = pred_tactile_values
                tactile_rec = tactile_transform(tactile_patches)

        else: 
            # splice out the mask tokens and project to pixel values
            if use_vision:
                mask_image_tokens = decoded_tokens[batch_range, masked_indices_image]
                pred_pixel_values = self.to_pixels(mask_image_tokens)
                image_patches[batch_range, masked_indices_image] = pred_pixel_values
                image_rec = image_transform(image_patches)

            # splice out the mask tokens and project to tactile values

            if self.num_tactiles > 0 and use_tactile:
                mask_tactile_tokens = decoded_tokens[batch_range, masked_indices_tactile]
                pred_tactile_values = self.to_tactiles(mask_tactile_tokens)
                tactile_patches[batch_range, masked_indices_tactile-image_patches.shape[1]] = pred_tactile_values
                tactile_rec = tactile_transform(tactile_patches)

            recon_loss_image = torch.tensor(0, device=device)
            recon_loss_tactile = torch.tensor(0, device=device)
            if use_vision:
                recon_loss_image = F.mse_loss(pred_pixel_values, masked_image_patches)
            if self.num_tactiles > 0 and use_tactile:
                recon_loss_tactile = F.mse_loss(pred_tactile_values, masked_tactile_patches)

        return_dict = {}
        if use_vision:
            return_dict['image_rec'] = image_rec
            return_dict['image_masked'] = image_masked
            return_dict['recon_loss_image'] = recon_loss_image
        if self.num_tactiles > 0 and use_tactile:
            return_dict['tactile_rec'] = tactile_rec
            return_dict['tactile_masked'] = tactile_masked
            return_dict['recon_loss_tactile'] = recon_loss_tactile

        return return_dict
    
    def get_embeddings(self, x, eval=True, use_vision=True, use_tactile=True):

        if eval:
            self.eval()
        else:
            self.train()

        if 'image' in x.keys():
            device = x['image'].device
        else:
            device = x['tactile1'].device
            use_vision = False

        # get patches
        if use_vision:
            image_patches = self.image_to_patch(x['image'])
            batch, num_image_patches, *_ = image_patches.shape
        else:
            image_patches = torch.zeros((x['tactile1'].shape[0], 0, 3)).to(device)
            num_image_patches = 0
        
        if self.num_tactiles > 0 and use_tactile:
            tactile_patches_list = []
            for i in range(1,self.num_tactiles+1):
                tactile_patches_list.append(self.tactile_to_patch(x['tactile'+str(i)]))
            tactile_patches = torch.cat(tactile_patches_list, dim=1)
            batch, num_tactile_patches, *_ = tactile_patches.shape
        else:
            tactile_patches = torch.zeros((x['image'].shape[0], 0, 3)).to(device)
            num_tactile_patches = 0
        
        num_patches = num_image_patches + num_tactile_patches

        # patch to encoder tokens and add positions

        if self.early_conv_masking:
            if use_vision:
                image_tokens = self.early_conv_vision(x['image'])
            else:
                image_tokens = torch.zeros((batch, 0, self.encoder_dim)).to(device)
            
            if self.num_tactiles > 0 and use_tactile:
                tactile_tokens_list = []
                for i in range(1,self.num_tactiles+1):
                    tactile_tokens_list.append(self.early_conv_tactile(x['tactile'+str(i)]))
                tactile_tokens = torch.cat(tactile_tokens_list, dim=1)
            else:
                tactile_tokens = torch.zeros((batch, 0, image_tokens.shape[-1])).to(device)
        else:
            if use_vision:
                image_tokens = self.image_patch_to_emb(image_patches)
            else:
                image_tokens = torch.zeros((batch, 0, self.encoder_dim)).to(device)

            if self.num_tactiles > 0 and use_tactile:
                tactile_tokens = self.tactile_patch_to_emb(tactile_patches)
            else:
                tactile_tokens = torch.zeros((batch, 0, image_tokens.shape[-1])).to(device)

        if use_vision:
            if self.use_sincosmod_encodings:
                image_tokens += self.encoder_modality_embedding(torch.tensor(0, device = device))
                image_tokens = image_tokens + self.image_enc_pos_embedding
        
        if self.num_tactiles > 0 and use_tactile:
            num_single_tactile_patches = num_tactile_patches//(self.num_tactiles)
            for i in range(self.num_tactiles):
                if self.use_sincosmod_encodings:
                    tactile_tokens[:, i*num_single_tactile_patches:(i+1)*num_single_tactile_patches] += self.encoder_modality_embedding(torch.tensor(1+i, device = device))
            if self.use_sincosmod_encodings:
                tactile_tokens = tactile_tokens + self.tactile_enc_pos_embedding

        tokens = torch.cat((image_tokens, tactile_tokens), dim=1)
        if not self.use_sincosmod_encodings:
            tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        return encoded_tokens

    def initialize_training(self, train_args):

        # training parameters
        lr = train_args['lr']
        
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.batch_size = train_args['batch_size']


    def train_iterations(self, iterations, replay_buffer, no_tactile=False):

        if len(replay_buffer) < self.batch_size:
            print("Not enough samples in replay buffer")
            return

        self.train()
    
        t = tqdm(range(iterations), desc='Iteration'.format(ncols=80))
        
        for i in t:
            
            x = random.choices(replay_buffer,k=self.batch_size)
            new_x = {}
            keys = ['image'] if no_tactile else ['image', 'tactile']
            for key in keys:
                new_x[key] = np.stack([x[j][key] for j in range(self.batch_size)])
            
            if 'image' in new_x:
                new_x['image'] = new_x['image'].transpose((0, 2, 3, 1, 4))
                new_x['image'] = new_x['image'].reshape((new_x['image'].shape[0], new_x['image'].shape[1], new_x['image'].shape[2], -1))
            if 'tactile' in new_x:
                new_x['tactile'] = new_x['tactile'].reshape((new_x['tactile'].shape[0], -1, new_x['tactile'].shape[3], new_x['tactile'].shape[4]))
            x = vt_load(new_x, frame_stack=self.frame_stack)

            if torch.cuda.is_available():
                for key in x:
                    x[key] = x[key].to('cuda')
            self.optimizer.zero_grad()
            r_loss = self(x)
            r_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            
            t.set_description("rloss: {}, lr: {}, Progress: ".format(r_loss, self.optimizer.param_groups[0]['lr']))

        self.eval()

class VTT(nn.Module):

    def __init__(self, *, image_size, tactile_size, image_patch_size, tactile_patch_size, dim, depth, heads, mlp_dim, image_channels = 3, tactile_channels=3, dim_head = 64, dropout = 0., emb_dropout = 0, num_tactiles=2, frame_stack=1):
        super().__init__()
        image_height, image_width = pair(image_size)
        tactile_height, tactile_width = pair(tactile_size)
        image_patch_height, image_patch_width = pair(image_patch_size)
        tactile_patch_height, tactile_patch_width = pair(tactile_patch_size)

        self.image_height = image_height
        self.image_width = image_width
        self.tactile_height = tactile_height
        self.tactile_width = tactile_width
        self.image_patch_height = image_patch_height
        self.image_patch_width = image_patch_width
        self.tactile_patch_height = tactile_patch_height
        self.tactile_patch_width = tactile_patch_width

        self.image_channels = image_channels
        self.tactile_channels = tactile_channels

        self.frame_stack = frame_stack

        assert image_height % image_patch_height == 0 and image_width % image_patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert tactile_height % tactile_patch_height == 0 and tactile_width % tactile_patch_width == 0, 'Tactile dimensions must be divisible by the patch size.'

        num_patches_image = (image_height // image_patch_height) * (image_width // image_patch_width)
        num_patches_tactile = (tactile_height // tactile_patch_height) * (tactile_width // tactile_patch_width) * num_tactiles

        num_patches = num_patches_image + num_patches_tactile

        image_patch_dim = image_channels * image_patch_height * image_patch_width
        tactile_patch_dim = tactile_channels * tactile_patch_height * tactile_patch_width
        
        self.image_to_patch_embedding = nn.Sequential(
            # Rearrange('b (n c) h w -> b c (n h) w', n = self.frame_stack, c = image_channels),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = image_patch_height, p2 = image_patch_width),
            nn.LayerNorm(image_patch_dim),
            nn.Linear(image_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.tactile_to_patch_embedding = nn.Sequential(
            # Rearrange('b (n c) h w -> b c (n h) w', n = self.frame_stack, c = tactile_channels),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = tactile_patch_height, p2 = tactile_patch_width),
            nn.LayerNorm(tactile_patch_dim),
            nn.Linear(tactile_patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

class MAEExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, mae_model, dim_embeddings, vision_only_control, frame_stack) -> None:
        super().__init__(observation_space, dim_embeddings)
        self.flatten = nn.Flatten()
        self.mae_model = mae_model
        
        self.running_buffer = {}

        self.vision_only_control = vision_only_control

        self.frame_stack = frame_stack
    
        self.vit_layer = VTT(
            image_size = (64, 64), # not used
            tactile_size = (32, 32), # not used
            image_patch_size = 8, # not used
            tactile_patch_size = 4, # not used
            dim = dim_embeddings, 
            depth = 1,
            heads = 4,
            mlp_dim = dim_embeddings*2,
            num_tactiles = 2, # not used
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        # print("image shape: ", observations['image'].shape)
        # print("tactile shape: ", observations['tactile'].shape)
        if 'image' in observations and len(observations['image'].shape) == 5:
            observations['image'] = observations['image'].permute(0, 2, 3, 1, 4)
            observations['image'] = observations['image'].reshape((observations['image'].shape[0], observations['image'].shape[1], observations['image'].shape[2], -1))
        if 'tactile' in observations and len(observations['tactile'].shape) == 5:
            observations['tactile'] = observations['tactile'].reshape((observations['tactile'].shape[0], -1, observations['tactile'].shape[3], observations['tactile'].shape[4]))
        
        # Get embeddings
        vt_torch = vt_load(observations, frame_stack=self.frame_stack)
        if torch.cuda.is_available():
            for key in vt_torch:
                vt_torch[key] = vt_torch[key].to('cuda')
        observations = self.mae_model.get_embeddings(vt_torch, eval=False, use_tactile=not self.vision_only_control)

        observations = self.vit_layer.transformer(observations)
        observations = torch.mean(observations, dim=1)
        
        flattened = self.flatten(observations)

        return flattened

class MAEPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        mae_model = None,
        dim_embeddings = 256,
        frame_stack = 1,
        vision_only_control = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
       

        features_extractor_class = MAEExtractor
        features_extractor_kwargs = {'mae_model': mae_model, 'dim_embeddings': dim_embeddings, 'vision_only_control': vision_only_control, 'frame_stack': frame_stack}
        ortho_init = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )

    def _get_latents(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract MAE/VIT features and split them into actor and critic latents.
        """
        features = self.extract_features(obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        return latent_pi, latent_vf

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self._get_latents(obs)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)

        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if embeddings.shape[-1] < self.dim:
            embeddings = F.pad(embeddings, (0, self.dim - embeddings.shape[-1]))

        return embeddings


class DiffusionActionHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_steps: int = 8,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        if num_steps < 1:
            raise ValueError("diffusion num_steps must be at least 1")

        self.num_steps = num_steps
        self.action_dim = action_dim

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.denoiser = nn.Sequential(
            nn.Linear(action_dim + latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.ones(1), alphas_cumprod[:-1]), dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    @staticmethod
    def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return buffer.gather(0, timesteps).reshape(-1, *([1] * (x.dim() - 1)))

    def predict_noise(self, noisy_action: torch.Tensor, timesteps: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        denoiser_input = torch.cat((noisy_action, latent, time_embedding), dim=-1)
        return self.denoiser(denoiser_input)

    def denoise(self, latent: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        batch_size = latent.shape[0]
        if deterministic:
            action = torch.zeros(batch_size, self.action_dim, device=latent.device, dtype=latent.dtype)
        else:
            action = torch.randn(batch_size, self.action_dim, device=latent.device, dtype=latent.dtype)

        for step in reversed(range(self.num_steps)):
            timesteps = torch.full((batch_size,), step, device=latent.device, dtype=torch.long)
            predicted_noise = self.predict_noise(action, timesteps, latent)

            beta_t = self._extract(self.betas, timesteps, action)
            sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, timesteps, action)
            sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, action)

            action = sqrt_recip_alpha_t * (action - beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t)
            if not deterministic and step > 0:
                variance_t = self._extract(self.posterior_variance, timesteps, action)
                action = action + torch.sqrt(variance_t) * torch.randn_like(action)

        return torch.tanh(action)


class DiffusionGaussianDistribution:
    def __init__(
        self,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ):
        self.mean_actions = mean_actions
        self.log_std = log_std.expand_as(mean_actions)
        self.std = torch.exp(self.log_std)
        self.distribution = torch.distributions.Normal(mean_actions, self.std)
        self.action_low = action_low
        self.action_high = action_high

    def _clip(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(actions, self.action_high), self.action_low)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            actions = self.mean_actions
        else:
            actions = self.distribution.rsample()
        return self._clip(actions)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.reshape_as(self.mean_actions)
        return self.distribution.log_prob(actions).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def mode(self) -> torch.Tensor:
        return self.mean_actions


class MAEDiffusionPolicy(MAEPolicy):
    """
    MAE/VIT feature extractor with a diffusion-shaped action head.

    PPO still needs a critic/value branch, so only the actor head is replaced.
    The diffusion head generates the mean action; a diagonal Gaussian wrapper
    supplies the log probability and entropy required by PPO.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = -1.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        mae_model = None,
        dim_embeddings = 256,
        frame_stack = 1,
        vision_only_control = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        diffusion_steps: int = 8,
        diffusion_hidden_dim: int = 256,
        diffusion_time_dim: int = 64,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 2e-2,
    ):
        if not isinstance(action_space, spaces.Box):
            raise ValueError("MAEDiffusionPolicy only supports continuous Box action spaces")
        if use_sde:
            raise ValueError("MAEDiffusionPolicy does not support gSDE; use the diffusion Gaussian wrapper instead")

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            mae_model,
            dim_embeddings,
            frame_stack,
            vision_only_control,
            optimizer_class,
            optimizer_kwargs,
        )

        self.action_dim = int(np.prod(action_space.shape))
        self.action_net = nn.Identity()
        if hasattr(self, "log_std"):
            nn.init.constant_(self.log_std, log_std_init)
        else:
            self.log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init)

        self.diffusion_actor = DiffusionActionHead(
            latent_dim=self.mlp_extractor.latent_dim_pi,
            action_dim=self.action_dim,
            hidden_dim=diffusion_hidden_dim,
            time_dim=diffusion_time_dim,
            num_steps=diffusion_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
        )

        self.register_buffer("action_low_tensor", torch.as_tensor(action_space.low, dtype=torch.float32).reshape(1, -1))
        self.register_buffer("action_high_tensor", torch.as_tensor(action_space.high, dtype=torch.float32).reshape(1, -1))

        optimizer_kwargs = self.optimizer_kwargs if hasattr(self, "optimizer_kwargs") else (optimizer_kwargs or {})
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **optimizer_kwargs)

    def _get_diffusion_distribution(self, latent_pi: torch.Tensor) -> DiffusionGaussianDistribution:
        mean_actions = self.diffusion_actor.denoise(latent_pi, deterministic=True)
        log_std = self.log_std.reshape(1, -1)
        return DiffusionGaussianDistribution(
            mean_actions=mean_actions,
            log_std=log_std,
            action_low=self.action_low_tensor,
            action_high=self.action_high_tensor,
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_pi, latent_vf = self._get_latents(obs)
        values = self.value_net(latent_vf)
        distribution = self._get_diffusion_distribution(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_pi, latent_vf = self._get_latents(obs)
        values = self.value_net(latent_vf)
        distribution = self._get_diffusion_distribution(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor) -> DiffusionGaussianDistribution:
        latent_pi, _ = self._get_latents(obs)
        return self._get_diffusion_distribution(latent_pi)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.forward(observation, deterministic=deterministic)[0]
