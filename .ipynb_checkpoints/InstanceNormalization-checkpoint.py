
import torch
from torch.nn import Module, Parameter


class QuaternionInstanceNorm2d(Module):
    r"""Applies a 2D Quaternion Instance Normalization to the incoming data.
        """

    def __init__(self, num_features, gamma_init=1., beta_param=True):
        super(QuaternionInstanceNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.eps = torch.tensor(1e-5)

    def reset_parameters(self):
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        quat_components = torch.chunk(input, 4, dim=1)
        
        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
        
       
        mu_r = torch.mean(r, dim=(2, 3), keepdim=True)
        mu_i = torch.mean(i, dim=(2, 3), keepdim=True)
        mu_j = torch.mean(j, dim=(2, 3), keepdim=True)
        mu_k = torch.mean(k, dim=(2, 3), keepdim=True)
        
        delta_r, delta_i, delta_j, delta_k = r - mu_r, i - mu_i, j - mu_j, k - mu_k
        
        
        quat_variance = torch.mean(delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2, dim=(2, 3), keepdim=True)
        denominator = torch.sqrt(quat_variance + self.eps)
    
        r_normalized = delta_r / denominator
        i_normalized = delta_i / denominator
        j_normalized = delta_j / denominator
        k_normalized = delta_k / denominator
        
        beta_components = torch.chunk(self.beta, 4, dim=1)
     
        new_r = (self.gamma * r_normalized) + beta_components[0]
        new_i = (self.gamma * i_normalized) + beta_components[1]
        new_j = (self.gamma * j_normalized) + beta_components[2]
        new_k = (self.gamma * k_normalized) + beta_components[3]
        
        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)
        
        return new_input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma.shape) \
               + ', beta=' + str(self.beta.shape) \
               + ', eps=' + str(self.eps.shape) + ')'
