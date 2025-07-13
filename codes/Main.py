import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import Functions
import warnings
warnings.filterwarnings("ignore")
# from tqdm.notebook import trange
from dataset.vos_dataset import VOSDataset
# import registration as R
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import json
import math

from LantentDynamics import ConvLSTM, PFLSTM
from VariationalAE import VariationalAutoEncoder
from Layers import VecInt, SpatialTransformer, LableSpatialTransformer, ConvLayer
from diffusion.model import LatentDiffusionmoprh
from Functions import NCC, Smooth, CrossEntropy, KullbackLeiblerDivergence, Dice, Hausdorff_distance, subplots
# import lightning
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  
# # torch.autograd.detect_anomaly()
# torch.cuda.device_count.cache_clear()

def load_data(data, name, mode='train'):

    info = data['info']
    datakeys = data.keys()

    if name == 'ACDC':

        if mode == 'val':         # for val
            idx = info[1]
            images = data['images'][2][:,idx[0]: idx[1]].cuda()
            frame_1st = data['images'][0].cuda()
            frame_xth = data['images'][1].cuda()
            if 'gts' in datakeys:
                frame_1st_gt = data['gts'][0].unsqueeze(dim=2).cuda()
                frame_xth_gt = data['gts'][1].unsqueeze(dim=2).cuda()
                gts = torch.cat([frame_1st_gt, frame_xth_gt], dim=2)
                la_1st, myo_1st, lv_1st = data['split_gts'][0]
                la_xth, myo_xth, lv_xth = data['split_gts'][1]
                split_gts = [la_1st, myo_1st, lv_1st, la_xth, myo_xth, lv_xth]
            else:
                gts, split_gts = None

            return info, images, frame_1st, frame_xth, gts, split_gts
                
        else:                              # for train 
            images = data['images'][0].cuda()

            return info, images, gts, split_gts
        
    else:
        info = data["info"]
        images = data['images'].cuda()    #[b, frame, slice, c, H, W]
        gts = data['gts'].cuda() if 'gts' in datakeys else None
        if 'gts' in datakeys:
            split_gts = data['split_gts']

        return info, images, gts, split_gts

class Network(nn.Module):

    def __init__(self, inshape, batch=1, reccurence=True, ssmver='prssm', data='anzhen'):
        super(Network, self).__init__()

        self.inshape = inshape
        self.batch = batch
        self.ssmver = ssmver
        self.hidden_dim = 4
        self.hiddim = 4
        self.reccurence = reccurence
        self.data = data

        if self.reccurence: 
            self.num_particals = 50
            self.Rnn = ConvLSTM(input_dim=self.hiddim, hidden_dim=self.hidden_dim, batch_size=batch)
            # self.Ssm = PFLSTM(num_particles=self.num_particals, 
            #                                input_size=inshape[0], 
            #                                hidden_size=self.hidden_dim, 
            #                                ext_act=64,
            #                                ext_obs=64,
            #                                resamp_alpha=0.5,)
        self.Vae = VariationalAutoEncoder(inshape, indim=1, hiddim=128)
        self.deform = LatentDiffusionmoprh(inshape, indim=4, diffuse='attnUnet')

        self.flow = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.integrate = VecInt(self.inshape, 7)
        self.transformer = SpatialTransformer(inshape, mode='bilinear')
        self.transformerlb = LableSpatialTransformer(inshape)
        
        # loss
        self.losses = {}
        self.pix_sim = nn.MSELoss(reduction='sum') if data not in [''] else NCC()
        # self.pix_sim = NCC()
        self.Smooth = Smooth().loss2d
        self.CrossEntropy = CrossEntropy().crossentropy
        self.kld_diffeo = KullbackLeiblerDivergence(prior_lambda=10).kld_loss
        self.kld_gauss = KullbackLeiblerDivergence(prior_lambda=10).kld_guassian
        self.dsc = Dice().loss
        self.dist = Hausdorff_distance().HausdorffDistance
        self.kl_weight = 1
        self.best_loss = None
        self.best_epoch = None

        self.case = None

    def latent_dynamics(self, data):
        
        if len(data.shape) == 5:
            T, S, C, H, W = data.shape
        elif len(data.shape) == 4:
            T, C, H, W = data.shape
            S = 1

        feat_m, feat_f, conditon = [], [], []
        input = data.reshape(T*S, C, H, W)

        z = self.Vae("uenc", input)
        z = self.Vae("udecx", z)
        # kld = self.kld_diffeo([mu, logvar])

        _,c,h,w = z.shape
        z = z.reshape(T, S, c, h, w)

        if self.reccurence:
            for t in range(T):
                f = z[t]
                m = z[0]

                # self.out_ssm = self.Ssm(f, self.out_ssm)  
                # self.out_rnn = self.Rnn(self.out_ssm[-1], self.out_rnn)
                self.out_rnn = self.Rnn(f, self.out_rnn)
                
                if t == 0:
                    self.hidden0 = self.out_rnn[0]   
                feat_m.append(self.hidden0)
                feat_f.append(self.out_rnn[0])
                # feat_m.append(self.out_rnn[0])
                # feat_f.append(self.hidden0)
                cond = torch.cat([data[0], data[t]], dim=-3) if len(data[0].shape) == 5 else torch.cat([data[0:1], data[t:t+1]], dim=-3)
                # cond = torch.cat([data[t], data[0]], dim=-3) if len(data[0].shape) == 5 else torch.cat([data[t:t+1], data[0:1]], dim=-3)
                conditon.append(cond)

            feat_m = torch.cat(feat_m, dim=0) # t,s,c,h,w
            feat_f = torch.cat(feat_f, dim=0)
            conditon = torch.cat(conditon, dim=0)
        # self.losses.update({'KLD': kld})

        return feat_m, feat_f, conditon
    
    def cal_deformation(self, f_m, f_f, conditon, mode='train'):

        if len(conditon.shape) == 5:
            t, s, c, h, w = conditon.shape
        elif len(conditon.shape) == 4:
            t, c, h, w = conditon.shape
            s = 1

        m = conditon.reshape(s*t, c, h, w)[:,0:1]
        f = conditon.reshape(s*t, c, h, w)[:,1:2]
        # subplots(m)
        if mode == 'train':
            z_pi, l_pix = self.deform.forward(f_m, f_f, conditon, mode='train')
            inf_Flow = self.integrate(z_pi)
            # subplots(inf_Flow[:,0]) 
            warp_moving = self.transformer(m, inf_Flow)
            neg_Flow = self.integrate(-z_pi)
            warp_fix = self.transformer(f, neg_Flow)
            losses = {'Score': l_pix,}
            self.losses.update(losses)
            return [inf_Flow, neg_Flow], [warp_moving, warp_fix]
            
        elif mode == 'val':
            # z_pi = self.deform.p_sample_loop(f_m, f_f, conditon)
            z_pi, l_pix = self.deform.forward(f_m, f_f, conditon)
            inf_Flow = self.integrate(z_pi)
            return inf_Flow
        
        elif mode == 'uncertain':
            pixel_wise_uncertainty, flows = self.deform.forward(f_m, f_f, conditon, mode=mode)

            return pixel_wise_uncertainty, flows

    def forward(self, input, gt=None, mode='val'):
        '''
        Moving frame: first frame;
        Fixed frame: other frames;
        inf_Flow: the deformation field from moving to fixed;
        neg_Flow: the deformation field from fixed to moving;
        warp_moving: the results of warping moving frame (pseudo fixed)
        warp_fix: the results of warping fixed frame (pseudo moving);
        '''

        if len(input.shape) == 6:
            b,t,s,c,h,w = input.shape
            input = input.reshape(t,b*s,c,h,w)
            gt = gt.reshape(t,b*s,h,w) if gt is not None else None
        elif len(input.shape) == 5:
            b,t,c,h,w = input.shape
            input = input.reshape(t,c,h,w)
            s=1
            
        self.initialisation(slice_num=s)
        self.moving = input[0:1].repeat(t,1,1,1,1)
        self.fixed = input
        # self.moving = input
        # self.fixed = input[0:1].repeat(t,1,1,1,1)
        if mode == 'train':
            feat_m, feat_f, conditon = self.latent_dynamics(input)
            flows, warping = self.cal_deformation(feat_m, feat_f, conditon)
            # subplots(self.moving)

            metrics = self.cal_metrics(flows, warping, gt)
            return self.losses, metrics
        
        elif mode == 'val':
            feat_m, feat_f, conditon = self.latent_dynamics(input)
            flows = self.cal_deformation(feat_m, feat_f, conditon, mode=mode)
            return flows
        
        elif mode == 'uncertain':
            feat_m, feat_f, conditon = self.latent_dynamics(input)
            pixel_wise_uncertainty, flows = self.cal_deformation(feat_m, feat_f, conditon, mode=mode)
            return pixel_wise_uncertainty, flows
        
    def cal_metrics(self, flows, warping=None, gt=None, mode='train'):
        '''
        flows[0]: inf_Flow;
        flows[1]: neg_Flow;
        gt: split gts
            ACDC: la_1st, myo_1st, lv_1st, la_xth, myo_xth, lv_xth
            else: myo, lv
            [b,s,h,w]
        '''
        if len(self.fixed.shape) == 5:
            t, s, c, h, w = self.fixed.shape
        elif len(self.fixed.shape) == 4:
            t, c, h, w = self.fixed.shape
            s = 1
        # a = self.pix_sim(self.moving[0,0,0], self.fixed[0,-1,0]).item()
        # subplots(self.fixed[0])
        losses = {}
        if mode == 'train':
            inf_smooth = self.Smooth(flows[0])
            inf_sim = self.pix_sim(warping[0], self.fixed.reshape(s*t,c,h,w))
            sim_before = self.pix_sim(self.moving.reshape(s*t,c,h,w), self.fixed.reshape(s*t,c,h,w)).item()
            neg_smooth = self.Smooth(flows[1])
            neg_sim = self.pix_sim(warping[1], self.moving.reshape(s*t,c,h,w))
            # alpha = 0.001 * (1 / (s*t)) if self.data not in [''] else 1
            alpha = 0.0001 if self.data not in [''] else 1
            bata = 10

            losses = {
            'inf_smooth': inf_smooth * bata,
            'inf_sim': inf_sim * alpha,
            'neg_smooth': neg_smooth * bata,
            'neg_sim': neg_sim * alpha,
            }

        elif mode == 'val':
            gt = [d.cuda() for d in gt]

            if len(gt) == 6:

                _, myo_1st, lv_1st, _, myo_xth, lv_xth = gt
                myo_gt_moving = (myo_1st + lv_1st) # 
                lv_gt_moving = lv_1st

                gt_moving = [myo_gt_moving, lv_gt_moving]
                gt_init = [myo_1st, lv_1st]
                gt_fixed = [myo_xth, lv_xth]
                flows = flows[-1]

            elif len(gt) == 2:

                myo, lv = gt
                if len(myo.shape) == 4:
                    b,t,h,w = myo.shape

                    myo_gt_moving = (myo[:,0] + lv[:,0]).repeat(1,t,1,1)
                    lv_gt_moving = lv[:,0].repeat(1,t,1,1)
                    gt_moving = [myo_gt_moving, lv_gt_moving]
                    gt_fixed = [myo.reshape(t,1,h,w), lv.reshape(t,1,h,w)]
                    gt_init = [myo[:,0].repeat(t,1,1,1), lv[:,0].repeat(t,1,1,1)]
                    flows = flows.reshape(t,2,h,w)

                elif len(myo.shape) == 5:
                    b,t,s,h,w = myo.shape

                    myo_gt_moving = (myo[:,0] + lv[:,0]).repeat(1,t,1,1,1)
                    lv_gt_moving = lv[:,0].repeat(1,t,1,1,1)

                    gt_moving = [myo_gt_moving, lv_gt_moving]
                    gt_fixed = [myo.reshape(t*s,1,h,w), lv.reshape(t*s,1,h,w)]
                    gt_init = [myo[:,0].repeat(t,1,1,1), lv[:,0].repeat(t,1,1,1)]
                    flows = flows.reshape(t,s,2,h,w)

            warp_msks = self.transformerlb.transforms(gt_moving, flows)
            # [Myo, Lv]
            warpping = [(warp_msks[0] - warp_msks[1]), warp_msks[1]]     # [b,t,s,c,h,w], where b refers to classes (myo, lv)
            # subplots((warp_msks[0] - warp_msks[1])[:,0])
            dsc_after, dsc_afte_temporal = self.dsc(warpping, gt_fixed)
            dsc_before, dsc_before_temporal = self.dsc(gt_init, gt_fixed)
            hd_after, mad_after = self.dist(warp_msks, gt_fixed)   # 0：endo 1：epi

        metrics = {
            'dsc_before': dsc_before,
            'dsc_after': dsc_after,
            'hd_after': hd_after,
            'mad_after': mad_after,
        } if gt is not None else {}

        self.losses.update(losses)

        return metrics

    def initialisation(self,slice_num):
        '''
        Initialise various param
        '''
        # Param of LSTM
        self.losses = {}

        if self.reccurence:
            h = int(self.inshape[0] / 4)
            w = int(self.inshape[0] / 4)
            self.hidden0 = torch.zeros(self.batch*slice_num, self.hidden_dim, h, w).cuda()
            self.h_rnn = torch.zeros(self.batch*slice_num, self.hidden_dim, h, w).cuda()
            self.c_rnn = torch.zeros(self.batch*slice_num, self.hidden_dim, h, w).cuda()
            self.out_rnn = [self.h_rnn, self.c_rnn, None]
            self.h_ssm = torch.zeros(self.batch*slice_num*self.num_particals, self.hidden_dim, h, w).cuda()
            self.c_ssm = torch.zeros(self.batch*slice_num*self.num_particals, self.hidden_dim, h, w).cuda()
            self.p_ssm = (torch.ones(self.batch*slice_num*self.num_particals, 1) * torch.log(torch.Tensor([1/self.num_particals]))).cuda()
            self.out_ssm = [self.h_ssm, self.c_ssm, self.p_ssm, None]

def save_model(model, optimizer, loss, best_loss, e, best_epoch, modality, name):
    save_path = '/data/zhuangshuxin/Codes/ASBIL-new/checkpoints'
    checkpoint = { 
        'info': 'Main',
        'loss': loss,
        'epoch': e,
        'network': model.state_dict(),
        'optimizer': optimizer.state_dict(),}

    if loss < best_loss:
        best_epoch = e
        best_loss = loss
        checkpoint_path = os.path.join(save_path, modality+'_'+name+'_best_checkpoint_modify.pth')
        torch.save(checkpoint, checkpoint_path)
    # print('best loss={} appears at {}, current epoch is {} and loss is {}'.format(best_loss, best_epoch, e, loss))
    else:
        checkpoint_path = os.path.join(save_path, modality+'_'+name+'_current_checkpoint_modify.pth')
        torch.save(checkpoint, checkpoint_path)
    print('best loss={} appears at {}, current epoch is {} and loss is {}'.format(best_loss, best_epoch, e, loss))
    return best_loss, best_epoch

def load_model(model, Optimizer=None, path=None):
    epoch = None
    Loss = None
    if path is not None:
        checkpoint = torch.load(path)
        keys = list(checkpoint.keys())
        network = checkpoint['network']
        for name, param in network.items():
            if name in model.state_dict():
                if param.size() == model.state_dict()[name].size():
                    model.state_dict()[name].copy_(param)
        network = {k.replace('module.', ''): v for k, v in network.items()}
        # model.load_state_dict(network)
        if Optimizer is not None:
            optimizer = checkpoint['optimizer'] 

            Optimizer.load_state_dict(optimizer)
        Loss = checkpoint['loss'] if 'loss' in keys else None
        epoch = checkpoint['epoch'] if 'epoch' in keys else None
    
    pre_ls = 999 if Loss is None else Loss
    best_loss = pre_ls
    best_epoch = epoch if epoch is not None else 0
    return epoch, pre_ls, best_loss, best_epoch

def train():

    checkpoints_path = f'/data/zhuangshuxin/Codes/ASBIL-new/checkpoints/2D_shengyi_ncc_current_checkpoint_modify.pth'
    name = 'shengyi'
    dim='2D'
    train_dataset = VOSDataset(name=name, mode='train', dim=dim, augmentation=False, interval=40)
    batchsize = 1
    vos_dataset = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize, 
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    # inshape = [160,160]
    inshape = [128,128]
    # inshape = [256,256]

    model = Network(inshape, int(batchsize), data=name).cuda()    #1e-3, no more
    Optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    pre_e, prev_ls, best_loss, best_epoch = load_model(model, path=checkpoints_path)
    total_epoch = range(2000) if pre_e == None else range(pre_e+1, 3000)
    # 训练
    model.train()
    print('Start training...')
    for e in total_epoch: 
        metrics = {}
        iter = 0
        Loss = 0
        for data in tqdm(vos_dataset):

            loss = 0
            info, images, gts, split_gts = load_data(data, name)

            cost, Metrics = model.forward(images, mode='train')

            if iter == 0:
                keys = list(Metrics.keys()) + list(cost.keys())
                metrics = {key : 0 for key in keys}
            for i, key in enumerate (cost.keys()):
                loss += torch.mean(cost[key])
                metrics[key] += torch.mean(cost[key]).item() / len(vos_dataset)
            for i, key in enumerate (Metrics.keys()):
                metrics[key] += Metrics[key] / len(vos_dataset)
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            iter+=1
            Loss += loss.item()

        best_loss, best_epoch = save_model(model, Optimizer, Loss/len(vos_dataset), best_loss, e, best_epoch, dim, name+'_ncc')
        print('ASBIL: ', metrics)
        if loss < prev_ls:
            prev_ls = loss.item()
        pass






if __name__ == "__main__":

    train()
    # val()

    # load_path = '/data1/zhuangshuxin/code/codes/3D_tracking/new_checkpoints/2D_Main_current_checkpoint_100p_PFSSM.pth'
    # model = Network100p([80,80], int(1)).cuda()    #1e-3, no more
    # pre_e, prev_ls = load_model(model, path= load_path)
    # v = Validation()
    # v.metrics(model, Optimizer=None)
        