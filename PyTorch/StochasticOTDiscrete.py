import numpy as np
import torch
import time


class PyTorchStochasticDiscreteTransport:


    def __init__(self, xs=None, ws=None, xt=None, wt=None, reg_type='entropy', reg_val=0.1):

        self.reg_type = reg_type
        self.reg_val = reg_val

        self.Xs = torch.from_numpy(xs)
        self.ws = torch.from_numpy(ws)

        self.Xt = torch.from_numpy(xt)
        self.wt = torch.from_numpy(wt)

        self.ns = xs.shape[0]
        self.nt = xt.shape[0]

        self.d = xt.shape[1]
        self.d_type = torch.float64

        self.u = torch.zeros(self.ns, dtype=self.d_type, requires_grad=True) # first dual variable
        self.v = torch.zeros(self.nt, dtype=self.d_type, requires_grad=True) # second dual variable


    def dual_OT_model(self, i_s, i_t):

        batch_size = i_s.shape[0]

        u_batch = torch.index_select(self.u, dim=0, index=i_s)
        v_batch = torch.index_select(self.v, dim=0, index=i_t)

        Xs_batch = torch.index_select(self.Xs, dim=0, index=i_s)
        Xt_batch = torch.index_select(self.Xt, dim=0, index=i_t)

        C_batch = torch.reshape(torch.sum(torch.mul(Xs_batch, Xs_batch), dim=1), (-1, 1)) + torch.reshape(torch.sum(torch.mul(Xt_batch, Xt_batch), dim=1), (1, -1)) - 2.*torch.matmul(Xs_batch, torch.transpose(Xt_batch, 0, 1)) # Cost matrix of the squared L2 norm ground cost

        if self.reg_type == 'entropy':

            loss_batch = torch.sum(u_batch)*batch_size + torch.sum(v_batch)*batch_size \
                         - self.reg_val*torch.sum(torch.exp((torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch) / self.reg_val))

        elif self.reg_type == 'l2':

            tmp = torch.max(0., (torch.reshape(u_batch, (-1, 1)) + torch.reshape(v_batch, (1, -1)) - C_batch))
            loss_batch = torch.sum(u_batch)*batch_size + torch.sum(v_batch)*batch_size \
                         - (1./(4.*self.reg_val))*torch.sum(torch.mul(tmp, tmp))

        return -loss_batch


    def learn_OT_dual_variables(self, epochs=10, batch_size=100, optimizer=None, lr=0.01, processor_type='cpu', processor_index="1"):

        if not optimizer:
            optimizer = torch.optim.SGD([self.u, self.v], lr=lr)

        batch_number_per_epoch = max([int((self.nt*self.nt)/float(batch_size*batch_size)), 1])

        self.losses = []
        self.time = []
        history = {}

        tic = time.time()

        for e in range(epochs):

            batch_losses = np.zeros((batch_number_per_epoch,))

            for b in range(batch_number_per_epoch):

                i_s = torch.from_numpy(np.random.choice(self.ns, size=(batch_size,), replace=False, p=self.ws)).type(torch.LongTensor)
                i_t = torch.from_numpy(np.random.choice(self.nt, size=(batch_size,), replace=False, p=self.wt)).type(torch.LongTensor)

                optimizer.zero_grad()

                loss_batch = self.dual_OT_model(i_s, i_t)
                loss_batch.backward()
                optimizer.step()

                batch_losses[b] = loss_batch.item()

            print('Epoch : {}, loss = {}'.format(e+1, np.sum(batch_losses)))

            self.losses.append(-np.sum(batch_losses)/(self.ns*self.nt))
            self.time.append(time.time()-tic)

        history['losses'] = self.losses
        history['time'] = self.time

        return history


    def compute_OT_MonteCarlo(self, epochs=10, batch_size=100): # before calling this, find the optimum dual variables with learn_OT_dual_variables
        batch_number_per_epoch = max([int((self.nt*self.nt)/float(batch_size*batch_size)), 1])
        OT_value = 0.
        for e in range(epochs):
            for b in range(batch_number_per_epoch):
                i_s = torch.from_numpy(np.random.choice(self.ns, size=(batch_size,), replace=False, p=self.ws)).type(torch.LongTensor)
                i_t = torch.from_numpy(np.random.choice(self.nt, size=(batch_size,), replace=False, p=self.wt)).type(torch.LongTensor)
                OT_value += self.dual_OT_model(i_s, i_t).item()
        return -OT_value/epochs/(self.nt*self.nt)

