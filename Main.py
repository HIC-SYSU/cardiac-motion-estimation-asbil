import os
import numpy as np
import torch
from tqdm import tqdm
from vos_dataset import VOSDataset
import torch.nn as nn
import torch.nn.functional as F
from LantentDynamics import dynamicLSTM, ConvLSTM
from Layers import VecInt, SpatialTransformer, NCC, Smooth, VariationalAutoEncoder
from Diffusion.model import Diffusion
import matplotlib.pyplot as plt
import math

class Network(nn.Module):

    def __init__(self, inshape):
        super(Network, self).__init__()

        self.inshape = inshape
        self.hidden_dim = 4
        self.hiddim = 4
        self.num_states = 50
        self.DRU = ConvLSTM(input_dim=self.hiddim, hidden_dim=self.hidden_dim)
        self.SRU = dynamicLSTM(num_states=self.num_states, 
                                input_size=inshape[0], 
                                hidden_size=self.hidden_dim, 
                                ext_act=64,
                                ext_obs=64,
                                resamp_alpha=0.5,)
        self.Vae = VariationalAutoEncoder(inshape, indim=1, hiddim=128)
        self.deform = Diffusion(inshape, indim=8)

        self.flow = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.integrate = VecInt(self.inshape, 7)
        self.transformer = SpatialTransformer(inshape, mode='bilinear')
        self.transformerlb = SpatialTransformer(inshape, mode='nearest')
        
        self.losses = {}
        self.pix_sim = nn.MSELoss(reduction='sum')
        # self.pix_sim = NCC()
        self.Smooth = Smooth().loss2d

    def latent_dynamics(self, data):
        
        T, C, H, W = data.shape

        feat_m, feat_f, conditon = [], [], []
        input = data

        z = self.Vae("uenc", input)
        z = self.Vae("udec", z)

        for t in range(T):
            f = z[t:t+1]
            
            self.out_dru = self.DRU(f, self.out_dru)  
            self.out_sru = self.SRU(self.out_dru[-1], self.out_sru)
            self.ht = torch.cat([self.out_dru[-1], self.out_sru[-1]], dim=1)
            
            if t == 0:
                self.h0 = self.ht
            feat_m.append(self.h0)
            feat_f.append(self.ht)
            cond = torch.cat([data[0], data[t]], dim=-3) if len(data[0].shape) == 5 else torch.cat([data[0:1], data[t:t+1]], dim=-3)
            conditon.append(cond)

        feat_m = torch.cat(feat_m, dim=0) # t,s,c,h,w
        feat_f = torch.cat(feat_f, dim=0)
        conditon = torch.cat(conditon, dim=0)

        return feat_m, feat_f, conditon
    
    def cal_deformation(self, f_m, f_f, conditon):

        m = conditon[:,0:1]
        f = conditon[:,1:2]

        z_pi, l_pix = self.deform.forward(f_m, f_f, conditon)
        inf_Flow = self.integrate(z_pi)

        warp_moving = self.transformer(m, inf_Flow)
        neg_Flow = self.integrate(-z_pi)
        warp_fix = self.transformer(f, neg_Flow)
        losses = {'Score': l_pix,}
        self.losses.update(losses)
        return [inf_Flow, neg_Flow], [warp_moving, warp_fix]

    def forward(self, input, gt=None):

        b,t,c,h,w = input.shape
        input = input.reshape(t,c,h,w)
            
        self.initialisation()
        self.moving = input[0:1].repeat(t,1,1,1)
        self.fixed = input

        feat_m, feat_f, conditon = self.latent_dynamics(input)
        flows, warping = self.cal_deformation(feat_m, feat_f, conditon)

        if gt is not None:
            init = gt[:,0].repeat(t,1,1,1).cuda()
            warp_gt = self.transformerlb(init.float(), flows[0])
        self.cal_metrics(flows, warping)
        return self.losses
        
    def cal_metrics(self, flows, warping):
        '''
        flows[0]: inf_Flow;
        flows[1]: neg_Flow;
        '''
        losses = {}
        inf_smooth = self.Smooth(flows[0])
        inf_sim = self.pix_sim(warping[0], self.fixed)
        pre_sim = self.pix_sim(self.moving, self.fixed)
        neg_smooth = self.Smooth(flows[1])
        neg_sim = self.pix_sim(warping[1], self.moving)
        alpha = 0.0001 
        bata = 10

        losses = {
        'inf_smooth': inf_smooth * bata,
        'inf_sim': inf_sim * alpha,
        'neg_smooth': neg_smooth * bata,
        'neg_sim': neg_sim * alpha,
        }

        self.losses.update(losses)

    def initialisation(self,):
        '''
        Initialise various param
        '''
        self.losses = {}

        h = int(self.inshape[0] / 4)
        w = int(self.inshape[0] / 4)
        self.h0 = torch.zeros(1, self.hidden_dim, h, w).cuda()
        self.h_dru = torch.zeros(1, self.hidden_dim, h, w).cuda()
        self.c_dru = torch.zeros(1, self.hidden_dim, h, w).cuda()
        self.out_dru = [self.h_dru, self.c_dru, None]
        self.h_sru = torch.zeros(self.num_states, self.hidden_dim, h, w).cuda()
        self.c_sru = torch.zeros(self.num_states, self.hidden_dim, h, w).cuda()
        self.p_sru = (torch.ones(self.num_states, 1) * torch.log(torch.Tensor([1/self.num_states]))).cuda()
        self.out_sru = [self.h_sru, self.c_sru, self.p_sru, None]

def load_model(model, Optimizer=None, path=None):
    checkpoint = torch.load(path)
    network = checkpoint['network']
    # for name, param in network.items():
    #     if name in model.state_dict():
    #         if param.size() == model.state_dict()[name].size():
    #             model.state_dict()[name].copy_(param)
    # network = {k.replace('module.', ''): v for k, v in network.items()}
    model.load_state_dict(network)
    if Optimizer is not None:
        optimizer = checkpoint['optimizer'] 
        Optimizer.load_state_dict(optimizer)


def train():

    checkpoints_path = r'/data/checkpoint.pth'

    train_dataset = VOSDataset(augmentation=False)
    batchsize = 1
    vos_dataset = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize, 
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    inshape = [128,128]

    model = Network(inshape).cuda()    #1e-3, no more
    Optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    load_model(model, Optimizer, path=checkpoints_path)
    total_epoch = range(200)
    model.train()

    for e in total_epoch: 
        metrics = {}
        iter = 0
        Loss = 0
        model.dice = 0
        for data in tqdm(vos_dataset):

            loss = 0
            info = data["info"]
            images = data['images'].cuda()    #[b, frame, slice, c, H, W]
            gts = data['gts'].cuda()
            # subplots(images)

            cost = model.forward(images, gts)
            if iter == 0:
                metrics = {key : 0 for key in list(cost.keys())}
            for i, key in enumerate(cost.keys()):
                loss += torch.mean(cost[key])
                metrics[key] += torch.mean(cost[key]).item() / len(vos_dataset)

            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            iter+=1
            Loss += loss.item()

        checkpoint_path = r'data/checkpoint.pth'
        checkpoint = {'network': model.state_dict(), 'optimizer': Optimizer.state_dict(),}
        torch.save(checkpoint, checkpoint_path)




if __name__ == "__main__":

    train()
