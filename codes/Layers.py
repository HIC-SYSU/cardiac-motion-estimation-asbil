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
