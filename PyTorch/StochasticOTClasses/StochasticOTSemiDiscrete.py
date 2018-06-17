import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as func
from StochasticOT import PyTorchStochasticOT


class PyTorchStochasticSemiDiscreteOT(PyTorchStochasticOT):


    def __init__(self, xt=None, wt=None, source_dual_variable_NN=None, reg_type='entropy', reg_val=0.1):

        PyTorchStochasticOT.__init__(self, reg_type=reg_type, reg_val=reg_val)

        self.Xt = torch.from_numpy(xt)
        self.wt = torch.from_numpy(wt)

        self.nt = xt.shape[0]
        self.d = xt.shape[1]

        if not source_dual_variable_NN:
            source_dual_variable_NN = Net(input_d=self.d, output_d=1)

        self.u = source_dual_variable_NN # first dual variable
        self.v = torch.zeros(self.nt, dtype=self.d_type, requires_grad=True) # second dual variable

        # In case we want to learn OT between the data and a Gaussian which is fitted to the data by Maximum Likelihood
        self.xt_mean = np.mean(xt, axis=0)
        self.gaussian_mean = self.xt_mean
        (U,S,V) = np.linalg.svd(np.cov(np.transpose(xt)))
        self.L = np.sqrt(S)[:, None]*V


    def dual_OT_model(self, Xs_batch, i_t):

        batch_size = i_t.shape[0]
        u_batch = self.u(Xs_batch)
        v_batch = torch.index_select(self.v, dim=0, index=i_t)
        Xt_batch = torch.index_select(self.Xt, dim=0, index=i_t)

        return self.dual_OT_batch_loss(batch_size=batch_size, u_batch=u_batch, v_batch=v_batch, Xs_batch=Xs_batch, Xt_batch=Xt_batch)


    def sampleFromGaussian(self, batch_size=100, mean=0., std=1.):
        xs_batch = mean + std * np.random.normal(size=(batch_size * self.d)).reshape((batch_size, self.d))
        Xs_batch = torch.from_numpy(xs_batch)
        return Xs_batch


    def sampleFromFittedGaussian(self, batch_size=100):
        xs_batch = self.gaussian_mean + np.dot(np.random.normal(size=(batch_size * self.d)).reshape((batch_size, self.d)), self.L)
        Xs_batch = torch.from_numpy(xs_batch)
        return Xs_batch


    def learn_OT_dual_variables(self, epochs=10, batch_size=100, source_sampling_function=None, optimizer=None, lr=0.01, device_type='cpu', device_index=0):

        if not source_sampling_function:
            source_sampling_function = self.sampleFromFittedGaussian

        trainable_params = list(Net().parameters())+[self.v]
        if not optimizer:
            optimizer = torch.optim.SGD(trainable_params, lr=lr)
        else :
            optimizer.add_group({'params' : trainable_params})

        batch_number_per_epoch = max([int((self.nt*self.nt)/float(batch_size*batch_size)), 1])
        losses = []
        times = []
        history = {}
        tic = time.time()

        for e in range(epochs):

            batch_losses = np.zeros((batch_number_per_epoch,))

            for b in range(batch_number_per_epoch):

                Xs_batch = source_sampling_function(batch_size=batch_size)
                i_t = torch.from_numpy(np.random.choice(self.nt, size=(batch_size,), replace=False, p=self.wt)).type(torch.LongTensor)

                optimizer.zero_grad()
                loss_batch = self.dual_OT_model(Xs_batch, i_t)
                loss_batch.backward()
                optimizer.step()

                batch_losses[b] = loss_batch.item()

            print('Epoch : {}, loss = {}'.format(e+1, np.sum(batch_losses)))

            losses.append(-np.sum(batch_losses)/(self.nt*self.nt))
            times.append(time.time()-tic)

        history['losses'] = losses
        history['time'] = times

        return history


    def compute_OT_MonteCarlo(self, epochs=10, batch_size=100, source_sampling_function=None, device_type='cpu', device_index=0): # before calling this, find the optimum dual variables with learn_OT_dual_variables
        if source_sampling_function == None:
            source_sampling_function = self.sampleFromFittedGaussian
        batch_number_per_epoch = max([int((self.nt*self.nt)/float(batch_size*batch_size)), 1])
        OT_value = 0.
        for e in range(epochs):
            for b in range(batch_number_per_epoch):
                Xs_batch = source_sampling_function(batch_size=batch_size)
                i_t = torch.from_numpy(np.random.choice(self.nt, size=(batch_size,), replace=False, p=self.wt)).type(torch.LongTensor)
                OT_value += self.dual_OT_model(Xs_batch, i_t).item()
        return -OT_value/epochs/(self.nt*self.nt)


    def barycentric_mapping_loss_model(self, neuralNet, Xs_batch, i_t):

        self.u.eval()
        self.v.requires_grad_(False)

        u_batch = self.u(Xs_batch)
        v_batch = torch.index_select(self.v, dim=0, index=i_t)
        Xt_batch = torch.index_select(self.Xt, dim=0, index=i_t)

        fXs_batch = neuralNet(Xs_batch)

        return self.barycentric_model_batch_loss(u_batch, v_batch, Xs_batch, Xt_batch, fXs_batch)


    def learn_barycentric_mapping(self, neuralNet=None, epochs=10, batch_size=100, source_sampling_function=None, optimizer=None, lr=0.01, device_type='cpu', device_index=0):

        if not source_sampling_function:
            source_sampling_function = self.sampleFromFittedGaussian

        if not neuralNet:
            neuralNet = Net(self.d, self.d)

        self.barycentric_mapping = neuralNet

        if not optimizer:
            optimizer = torch.optim.SGD(neuralNet.parameters(), lr=lr)

        batch_number_per_epoch = max([int((self.nt*self.nt)/float(batch_size*batch_size)), 1])

        barycentric_mapping_losses = []
        barycentric_mapping_time = []
        history = {}

        tic = time.time()

        for e in range(epochs):

            batch_losses = np.zeros((batch_number_per_epoch,))

            for b in range(batch_number_per_epoch):

                Xs_batch = source_sampling_function(batch_size=batch_size)
                i_t = torch.from_numpy(np.random.choice(self.nt, size=(batch_size,), replace=False, p=self.wt)).type(torch.LongTensor)

                optimizer.zero_grad()

                loss_batch = self.barycentric_mapping_loss_model(neuralNet, Xs_batch, i_t)
                loss_batch.backward()
                optimizer.step()

                batch_losses[b] = loss_batch.item()

            print('Epoch : {}, barycentric_mapping loss = {}'.format(e+1, np.sum(batch_losses)))

            barycentric_mapping_losses.append(np.sum(batch_losses)/(self.nt*self.nt))
            barycentric_mapping_time.append(time.time()-tic)

        history['losses'] = barycentric_mapping_losses
        history['time'] = barycentric_mapping_time

        return history



class Net(nn.Module):
    def __init__(self, input_d=2, output_d=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_d, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_d)

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.fc4(x).type(torch.DoubleTensor)
        return x


