import torch.nn as nn
import torch.nn.functional as F
from Layers import ConvLayer
import torch

#==============================================
# ConvLSTM
#==============================================
class ConvLSTMCell(nn.Module):

    def __init__(self, device, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, slc_n=9):
        height, width = image_size
        return (torch.zeros(slc_n, self.hidden_dim, height, width, device=self.conv.weight.device).cuda(self.device),
                torch.zeros(slc_n, self.hidden_dim, height, width, device=self.conv.weight.device).cuda(self.device))

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
            [bactsize,seq_length,channel,H,W]
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers): #only 1

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]) 
                #cell_list为cell模型
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
class ConvLSTMCell_para(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell_para, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = F.leaky_relu(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM_para(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, kernel_size=3, num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM_para, self).__init__()

        # self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        # hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        # if not len(kernel_size) == len(hidden_dim) == num_layers:
        #     raise ValueError('Inconsistent list length.')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cur_input_dim = self.input_dim
        self.cell = ConvLSTMCell_para(input_dim=cur_input_dim,
                                    hidden_dim=self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    bias=self.bias,)
        # self.hidden_state = self._init_hidden(batch_size=batch_size, image_size=(80, 80), slc_n=9)
        # self.h, self.c = self.hidden_state
    
    def forward(self, input_tensor, hx):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
            [bactsize,seq_length,channel,H,W]
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if len(hx)==2:
            h, c = hx
        else:
            h, c, _ = hx
        h, c = self.cell(input_tensor=input_tensor, cur_state=[h, c])
        return h, c  # shape (b, slc, c, h, w)
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

class PFRNNBaseCell(nn.Module):
    """
    This is the base class for the PF-RNNs. We implement the shared functions here, including
        1. soft-resampling
        2. reparameterization trick
        3. obs_extractor o_t(x_t)
        4. control_extractor u_t(x_t)
        All particles in PF-RNNs are processed in parallel to benefit from GPU parallelization.
    """

    def __init__(self, num_particles, input_size, hidden_size, ext_obs, ext_act, resamp_alpha):
        """
        :param num_particles: number of particles for a PF-RNN
        :param input_size: the size of input x_t
        :param hidden_size: the size of the hidden particle h_t^i
        :param ext_obs: the size for o_t(x_t)
        :param ext_act: the size for u_t(x_t)
        :param resamp_alpha: the control parameter \alpha for soft-resampling.
        We use the importance sampling with a proposal distribution q(i) = \alpha w_t^i + (1 - \alpha) (1 / K)
        """
        super(PFRNNBaseCell, self).__init__()
        self.num_particles = num_particles
        self.input_size = input_size
        self.h_dim = hidden_size
        self.ext_obs = ext_obs
        self.ext_act = ext_act
        self.resamp_alpha = resamp_alpha

        self.obs_extractor = nn.Conv2d(self.input_size, self.ext_obs, kernel_size=3, padding=1)
        self.act_extractor = nn.Conv2d(self.input_size, self.ext_act, kernel_size=3, padding=1)
        
        # nn.Sequential(
        #     nn.Conv2d(self.input_size, self.ext_obs, kernel_size=3, padding=1),
        #     nn.LeakyReLU()
        # )
        # self.act_extractor = nn.Sequential(
        #     nn.Conv2d(self.input_size, self.ext_act, kernel_size=3, padding=1),
        #     nn.LeakyReLU()
        # )

        self.conv_obs = nn.Conv2d(2 * self.h_dim, 1, kernel_size=3, padding=1)

        # self.fc_obs = nn.Linear(32*32, 1)
        self.fc_obs = nn.Linear(64*64, 1)

        # self.batch_norm = nn.BatchNorm2d(self.num_particles)

    def resampling(self, particles, prob):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.
        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [num_particles * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 -self.resamp_alpha) * 1 / self.num_particles # equation 3
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1), num_samples=self.num_particles, replacement=True)
        # print(resamp_prob)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor)
        offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        particles_new = (particles[0][flatten_indices],particles[1][flatten_indices]) if len(particles) == 2 else \
                            particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 - self.resamp_alpha) / self.num_particles)     # equation 3
        prob_new = torch.log(prob_new).view(self.num_particles, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new

class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, kernel_size=3, num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False, mode='nonlinear'):
        super(ConvLSTM, self).__init__()

        self.mode = mode
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        if self.mode == 'nonlinear':
            self.cell = ConvLSTMCell_para(input_dim=self.input_dim,
                                        hidden_dim=self.hidden_dim,
                                        kernel_size=self.kernel_size,
                                        bias=self.bias,)
        elif self.mode == 'linear':
            self.cell = nn.Sequential(
                            nn.Conv2d(2*self.input_dim, self.hidden_dim, kernel_size=3, padding=1),
                                    )
            
    
    def forward(self, input_tensor, hx):
        if len(hx)==1:
            h = hx[0]
        elif len(hx)==2:
            h, c = hx
        elif len(hx)==3:
            h, c, _ = hx

        if self.mode == 'nonlinear':
            h, c = self.cell(input_tensor=input_tensor, cur_state=[h, c])
            return h, c  # shape (b, slc, c, h, w)
        elif self.mode == 'linear':
            h = self.cell(torch.cat((input_tensor, h), dim=1))
            return [h]  # shape (b, slc, c, h, w)

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class PFLSTM(PFRNNBaseCell):
    def __init__(self, num_particles, input_size, hidden_size, ext_obs, ext_act, resamp_alpha, mode='nonlinear'):
        super().__init__(num_particles, input_size, hidden_size, ext_obs, ext_act, resamp_alpha)
        self.mode = mode
        if self.mode == 'nonlinear':
            self.conv = nn.Conv2d(2*self.h_dim, 4 * self.h_dim, kernel_size=3, padding=1)
        elif self.mode == 'linear':
            self.conv = nn.Sequential(
                        nn.Conv2d(2*self.h_dim, self.h_dim, kernel_size=3, padding=1),
                        )
        self.conv_obs = nn.Conv2d(self.h_dim, 1, kernel_size=3, padding=1)

        w = int(input_size/4)
        self.fc_obs = nn.Linear(w*w, 1)
        

    def forward(self, input, hx):
        
        h0, c0, p0, _ = hx
        b,c,h,w = input.shape
        # generate particles
        input = input.repeat(self.num_particles,1,1,1,1).view(b*self.num_particles,c,h,w)   
        s = torch.cat([input, h0], dim=1)

        if self.mode == 'nonlinear':
            s = self.conv(s)
            f, i, o, g = torch.split(s, split_size_or_sections=self.h_dim, dim=1)
            c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * F.leaky_relu(g)
            h1 = torch.sigmoid(o) * torch.tanh(c1)
            att = torch.cat((input, h1), dim=1)

        elif self.mode == 'linear':
            att = self.conv(s)

        # calculating the log-form weight of the particles
        logpdf_obs = self.conv_obs(att)
        logpdf_obs = self.fc_obs(logpdf_obs.view(b*self.num_particles, h*w))
        p1 = logpdf_obs.view(self.num_particles, -1, 1) * p0.view(self.num_particles, -1, 1)
        # a = torch.logsumexp(p1, dim=0, keepdim=True)
        p1 = p1 - torch.logsumexp(p1, dim=0, keepdim=True)  # normalization
        # resample particles
        if self.mode == 'nonlinear':
            (h1, c1), p1 = self.resampling((h1, c1), p1) 
        elif self.mode == 'linear':
            a = torch.unique(p1)
            h1, p1 = self.resampling(att, p1)
            c1 = None

        _,ch,rol,cal = h1.shape
        hidden = h1.view(b,self.num_particles,-1)* torch.exp(p1.view(b,self.num_particles,1))
        hidden = torch.sum(hidden.view(b,self.num_particles,ch,rol,cal), dim=1)

        return h1, c1, p1, hidden