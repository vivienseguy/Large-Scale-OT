import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as func


class PyTorchStochasticOT:


    def __init__(self, reg_type='entropy', reg_val=0.1):

        self.reg_type = reg_type
        self.reg_val = reg_val
        self.d_type = torch.float64

        self.barycentric_mapping = None


    def computeSquareEuclideanCostMatrix(self, Xs_batch, Xt_batch):
        return torch.reshape(torch.sum(torch.mul(Xs_batch, Xs_batch), dim=1), (-1, 1)) + torch.reshape(torch.sum(torch.mul(Xt_batch, Xt_batch), dim=1), (1, -1)) \
               - 2. * torch.matmul(Xs_batch,torch.transpose(Xt_batch, 0,1))


    def dual_OT_batch_loss(self, batch_size, u_batch, v_batch, Xs_batch, Xt_batch):

        C_batch = self.computeSquareEuclideanCostMatrix(Xs_batch=Xs_batch, Xt_batch=Xt_batch)

        if self.reg_type == 'entropy':
            loss_batch = torch.sum(u_batch)*batch_size + torch.sum(v_batch)*batch_size \
                         - self.reg_val*torch.sum(torch.exp((torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch) / self.reg_val))
        elif self.reg_type == 'l2':
            tmp = torch.max(torch.zeros(1).type(torch.DoubleTensor), (torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch))
            loss_batch = torch.sum(u_batch)*batch_size + torch.sum(v_batch)*batch_size \
                         - (1./(4.*self.reg_val))*torch.sum(torch.mul(tmp, tmp))

        return -loss_batch


    def barycentric_model_batch_loss(self, u_batch, v_batch, Xs_batch, Xt_batch, fXs_batch):

        C_batch = self.computeSquareEuclideanCostMatrix(Xs_batch, Xt_batch)

        if self.reg_type == 'entropy':
            H = torch.exp((torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch) / self.reg_val)
        elif self.reg_type == 'l2':
            H = (1./(2.*self.reg_val))*torch.max(torch.zeros(1).type(torch.DoubleTensor), (torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch))

        d_batch = self.computeSquareEuclideanCostMatrix(fXs_batch, Xt_batch)

        return torch.sum(torch.mul(d_batch, H))


    def evaluate_barycentric_mapping(self, xs):
        self.barycentric_mapping.eval()
        xs_tensor = torch.from_numpy(xs)
        return self.barycentric_mapping(xs_tensor).detach().numpy()




