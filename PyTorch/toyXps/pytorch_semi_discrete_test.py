#%% 2-dimensional test script for the computation of regularized OT between a Gaussian and a discrete data set (semi-discrete OT)
import numpy as np
import matplotlib.pylab as pl
import torch
from PyTorch.StochasticOTClasses.StochasticOTSemiDiscrete import PyTorchStochasticSemiDiscreteOT


reg_type = 'l2'
reg_val = 0.05
device_type = 'cpu'
device_index = 0

# Initialize target discrete data set
d=2
mean=0.
std=1.
def gaussianSamplingFunction(batch_size=100):
    xs_batch = mean + std*np.random.normal(size=(batch_size * d)).reshape((batch_size, d))
    Xs_batch = torch.from_numpy(xs_batch)
    return Xs_batch

nt=200
scale = 2.5
xt = np.concatenate((np.random.choice(np.array([-scale, scale]), nt).reshape((-1,1)), np.random.choice(np.array([-scale, scale]), nt).reshape((-1,1))),1)
xt = xt+0.2*np.random.randn(nt,2)
wt = np.ones((nt,))/nt

# Dual OT Stochastic Optimization (alg.1 of ICLR 2018 paper "Large-Scale Optimal Transport and Mapping Estimation")
semiDiscreteOTComputer = PyTorchStochasticSemiDiscreteOT(xt, wt, source_dual_variable_NN=None, reg_type=reg_type, reg_val=reg_val)
history = semiDiscreteOTComputer.learn_OT_dual_variables(epochs=1000, batch_size=50, source_sampling_function=gaussianSamplingFunction, lr=0.0001, device_type=device_type, device_index=device_index)

pl.figure(1)
pl.plot(history['losses'], lw=3, label='loss')
pl.legend(loc='best')
pl.title('Semi-Discrete: Loss per epoch')
pl.savefig('semi_discrete_loss_per_epoch.png')


# Learn Barycentric Mapping (alg.2 of ICLR 2018 paper "Large-Scale Optimal Transport and Mapping Estimation")
bp_history = semiDiscreteOTComputer.learn_barycentric_mapping(epochs=200, batch_size=50, source_sampling_function=gaussianSamplingFunction, lr=0.00001, device_type=device_type, device_index=device_index)

pl.figure(2)
pl.plot(bp_history['losses'], lw=3, label='loss')
pl.title('Loss per epochs')
pl.savefig('semi_discrete_barycentric_map_learning_loss.png')


# Compute Mapping estimation
xs = gaussianSamplingFunction(10*nt).numpy()
xsf = semiDiscreteOTComputer.evaluate_barycentric_mapping(xs)

pl.figure(3)
pl.plot(xs[:, 0], xs[:, 1], '+b')
pl.plot(xt[:, 0], xt[:, 1], 'xr')
pl.plot(xsf[:, 0], xsf[:, 1], '+g')
pl.savefig('semi_discrete_mapping_estimation.png')

