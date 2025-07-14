import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
                    # [x,x]
        super().__init__()

        self.mode = mode
        self.size = size
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors) # 生成网格（代表x,y的矩阵）
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0) #[1,2,x,x]
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):

        if not isinstance(src, float):
            src = src.float()
        new_locs = self.grid.cuda() + flow
        shape = flow.shape[2:]  # [h,w]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)  #归一化
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
                                # 7
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec
    
class LableSpatialTransformer():

    def __init__(self, inshape):
        self.inshape = inshape
        self.transformer = SpatialTransformer(self.inshape, mode='nearest')

    def transforms(self, label, flow):
        '''
        
        '''
        if not isinstance(label, list):
            raise ValueError('label should be a list.')
        
        if len(flow.shape) == 5:
            t,s,c,h,w = flow.shape
            
            warping = []
            for lb in label:
                b = lb.shape[0]
                temporal = []
                init_lb = lb.reshape(t, b*s, 1, h, w)

                for i in range(t):
                    temporal.append(self.transformer(init_lb[i], flow[i]))
                warping.append(torch.stack(temporal, dim=0).squeeze())

        elif len(label[0].shape) == 4:
            n,c,h,w = flow.shape

            warping = []
            for lb in label:
                b = lb.shape[0]
                lb = lb.reshape(b*n, 1, h, w)
                warping.append(self.transformer(lb, flow).squeeze())

        return warping

class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=9, eps=1e-3):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 2
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        # prepare conv kernel
        conv_fn = getattr(F, 'conv%dd' % ndims)
        # conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        # win_size = np.prod(self.win)
        win_size = torch.from_numpy(np.array([np.prod(self.win)])).float()
        win_size = win_size.cuda()
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc0 = cross * cross / (I_var * J_var + self.eps)
        cc = torch.clamp(cc0, 0.001, 0.999)

        # return negative cc.
        return -1.0 * torch.mean(cc)
    
class Smooth:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss3d(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

    def loss2d(self, s, penalty='l2'):
        # s is the deformation_matrix of shape (seq_length, channels=2, height, width)
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

class VariationalAutoEncoder(nn.Module):

    def __init__(self, inshape, indim, hiddim, appdim=None):
        super().__init__()  
        
        self.inshape = inshape
        self.indim = indim
        self.hiddim = hiddim
        self.appdim = appdim
        self.unet = Unet(self.indim, self.hiddim, self.appdim)

        self.mu = nn.Conv2d(4, 4, 3, padding=1)
        self.var = nn.Conv2d(4, 4, 3, padding=1)

    
    def uenc(self, input):

        out = self.unet('encoder', input)

        return out
    
    def udec(self, input, modes=None):

        x = self.unet('decoder', input, modes=modes)

        return x
    
    def forward(self, mode, *args, **kwargs):

        if mode == 'uenc':
            return self.uenc(*args, **kwargs)
        elif mode == 'udec':
            return self.udec(*args, **kwargs)
        else:
            raise NotImplementedError
        

class Unet(nn.Module):
    def __init__(self, indim, hiddim, appdim=None, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()

        ndims = 2
        self.appdim = appdim
        self.hiddim = hiddim
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
                nb_features = [
                                [32, 64, 128, self.hiddim],             # encoder
                                [self.hiddim, 128, 64, 32, 16, 8, 4]  # decoder
                            ]
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features  

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.kernal_size = [3, 3, 3, 3]
        self.padding = [1, 1, 1, 1]
        self.stride = [2, 2, 2, 2]

        prev_nf = 1
        self.downarm = nn.ModuleList()
        self.downarm.append(ConvBlock(ndims, indim, self.enc_nf[0], 7, stride=1, padding=3))
        prev_nf = self.enc_nf[0]
        for i in range(len(self.enc_nf)):
            self.downarm.append(ConvBlock(ndims, prev_nf, self.enc_nf[i], self.kernal_size[i], stride=self.stride[i], padding=self.padding[i]))    
            prev_nf = self.enc_nf[i]       

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            # channels = prev_nf + enc_history[i]*2 if i > 0 else prev_nf*2
            if self.appdim is None:
                channels = prev_nf
            else:
                channels = prev_nf + self.appdim if i == 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, 3, stride=1, padding=1))
            prev_nf = nf
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, 3, stride=1, padding=1))
            prev_nf = nf

    def encoder(self, x):
        x_enc = [x] 
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        # conv, upsample, concatenate series
        return x_enc[-1]
    
    def decoder(self, x, modes=None):
        for i, layer in enumerate(self.uparm):
            x = layer(x)
            if modes == 'upsampling':
                x = self.upsample(x)
            else:
                x = self.upsample(x) if i%2==0 else x
        for layer in self.extras:
            x = layer(x)
        return x   
        
    def forward(self, mode, *args, **kwargs):
        if mode == 'encoder':
            return self.encoder(*args, **kwargs)
        elif mode == 'decoder':
            return self.decoder(*args, **kwargs) 
        else:
            raise NotImplementedError

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        dillation = 1
        self.main = Conv(in_channels, out_channels, kernel_size, stride, padding, dillation)
        self.activation = nn.LeakyReLU(0.2)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.main(x)
        out = self.batchnorm(out)
        out = self.activation(out)
        return out