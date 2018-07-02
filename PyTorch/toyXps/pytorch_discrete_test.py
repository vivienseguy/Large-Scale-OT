import numpy as np
import matplotlib.pylab as pl
import sys
from PyTorch.StochasticOTClasses.StochasticOTDiscrete import PyTorchStochasticDiscreteOT



# Initialize data
d = 2
ns = 300
nt = 300
reg_val = 0.02
reg_type = 'l2'
device_type = 'cpu'
device_index = 0

radius = 1.
noise_a = 0.1
t = np.random.rand(ns)*2*np.pi
xs = np.concatenate((np.cos(t).reshape((-1,1)), np.sin(t).reshape((-1,1))),1)
xs = radius*xs + noise_a*np.random.randn(ns,2)
ws = np.ones((ns,))/ns

radius = 2.
noise_a = 0.2
t = np.random.rand(nt)*2*np.pi
xt = np.concatenate((np.cos(t).reshape((-1,1)),np.sin(t).reshape((-1,1))),1)
xt = radius*xt + noise_a*np.random.randn(nt,2)
wt = np.ones((nt,))/nt


# Dual OT Stochastic Optimization (alg.1 of ICLR 2018 paper "Large-Scale Optimal Transport and Mapping Estimation")
discreteOTComputer = PyTorchStochasticDiscreteOT(xs, ws, xt, wt, reg_type, reg_val, device_type=device_type, device_index=device_index)
history = discreteOTComputer.learn_OT_dual_variables(epochs=1000, batch_size=50, lr=0.0005)

pl.figure(1)
pl.plot(history['losses'], lw=3, label='loss')
pl.legend(loc='best')
pl.title('Loss per epoch')
pl.savefig('loss_per_epoch.png')


# Compute the reg-OT objective
d_stochastic = discreteOTComputer.compute_OT_MonteCarlo(epochs=20, batch_size=50)
print('dual objective: %f\n' % d_stochastic)


# Learn Barycentric Mapping (alg.2 of ICLR 2018 paper "Large-Scale Optimal Transport and Mapping Estimation")
bp_history = discreteOTComputer.learn_barycentric_mapping(epochs=300, batch_size=50, lr=0.000002)

pl.figure(2)
pl.plot(bp_history['losses'], lw=3, label='loss')
pl.title('Loss per epochs')
pl.savefig('barycentric_map_learning_loss.png')

# Compute Mapping estimation
xsf = discreteOTComputer.evaluate_barycentric_mapping(xs)

pl.figure(3)
pl.plot(xs[:, 0], xs[:, 1], '+b', label='source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='target samples')
pl.plot(xsf[:, 0], xsf[:, 1], '+g', label='mapped source samples')
pl.legend()

pl.savefig('mapping_estimation.png')
