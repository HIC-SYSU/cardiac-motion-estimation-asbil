import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

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
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = F.leaky_relu(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1,
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

        self.cell = ConvLSTMCell(input_dim=self.input_dim,
                                    hidden_dim=self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    bias=self.bias,)
        

    def forward(self, input_tensor, hx):
        if len(hx)==1:
            h = hx[0]
        elif len(hx)==2:
            h, c = hx
        elif len(hx)==3:
            h, c, _ = hx

        h, c = self.cell(input_tensor=input_tensor, cur_state=[h, c])
        return h, c



class dynamicBaseCell(nn.Module):

    def __init__(self, num_states, input_size, hidden_size, ext_obs, ext_act, resamp_alpha):
 
        super(dynamicBaseCell, self).__init__()
        self.num_states = num_states
        self.input_size = input_size
        self.h_dim = hidden_size
        self.ext_obs = ext_obs
        self.ext_act = ext_act
        self.resamp_alpha = resamp_alpha

        self.obs_extractor = nn.Conv2d(self.input_size, self.ext_obs, kernel_size=3, padding=1)
        self.act_extractor = nn.Conv2d(self.input_size, self.ext_act, kernel_size=3, padding=1)

        self.conv_obs = nn.Conv2d(2 * self.h_dim, 1, kernel_size=3, padding=1)

        self.fc_obs = nn.Linear(64*64, 1)

        self.batch_norm = nn.BatchNorm2d(self.num_states)

    def resampling(self, states, prob):
 
        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 -self.resamp_alpha) * 1 / self.num_states # equation 3
        resamp_prob = resamp_prob.view(self.num_states, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1), num_samples=self.num_states, replacement=True)
        # print(resamp_prob)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor)
        offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        new_states = (states[0][flatten_indices],states[1][flatten_indices]) if len(states) == 2 else \
                            states[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 - self.resamp_alpha) / self.num_states)     # equation 3
        prob_new = torch.log(prob_new).view(self.num_states, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return new_states, prob_new

class dynamicLSTM(dynamicBaseCell):
    def __init__(self, num_states, input_size, hidden_size, ext_obs, ext_act, resamp_alpha):
        super().__init__(num_states, input_size, hidden_size, ext_obs, ext_act, resamp_alpha)
        self.conv = nn.Conv2d(2*self.h_dim, 4 * self.h_dim, kernel_size=3, padding=1)
        self.conv_obs = nn.Conv2d(self.h_dim, 1, kernel_size=3, padding=1)

        w = int(input_size/4)
        self.fc_obs = nn.Linear(w*w, 1)
        

    def forward(self, input, hx):
        
        h0, c0, p0, _ = hx
        b,c,h,w = input.shape
        # creating substates
        input = input.repeat(self.num_states,1,1,1,1).view(b*self.num_states,c,h,w)   
        s = torch.cat([input, h0], dim=1)

        s = self.conv(s)
        f, i, o, g = torch.split(s, split_size_or_sections=self.h_dim, dim=1)
        c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * F.leaky_relu(g)
        h1 = torch.sigmoid(o) * torch.tanh(c1)
        att = torch.cat((input, h1), dim=1)

        logpdf_obs = self.conv_obs(att)
        logpdf_obs = self.fc_obs(logpdf_obs.view(b*self.num_states, h*w))
        p1 = logpdf_obs.view(self.num_states, -1, 1) * p0.view(self.num_states, -1, 1)
        p1 = p1 - torch.logsumexp(p1, dim=0, keepdim=True)  # normalization

        (h1, c1), p1 = self.resampling((h1, c1), p1) 

        _,ch,rol,cal = h1.shape
        hidden = h1.view(b,self.num_states,-1)* torch.exp(p1.view(b,self.num_states,1))
        hidden = torch.sum(hidden.view(b,self.num_states,ch,rol,cal), dim=1)

        return h1, c1, p1, hidden