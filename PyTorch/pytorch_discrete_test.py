import numpy as np
import matplotlib.pylab as pl
import ot
from StochasticOTDiscrete import PyTorchStochasticDiscreteTransport

n = 100
d = 2
ns = 60
nt = 60
reg_val = 0.1

nz = 2.
t = np.random.rand(ns)*2*np.pi
xs = np.concatenate((np.cos(t).reshape((-1,1)), np.sin(t).reshape((-1,1))),1)
ws = np.ones((ns,))/ns

t = np.random.rand()*2*np.pi
xt = np.concatenate((np.cos(t).reshape((-1,1)),np.sin(t).reshape((-1,1))),1)
xt = xt*2. + nz*np.random.randn(nt,2)
wt = np.ones((nt,))/nt

# pl.figure(1)
# pl.clf()
# pl.subplot(1,2,1)
# pl.hist2d(xs[:,0],xs[:,1],bins=n,range=[[-3,3],[-3,3]],cmap='Blues')
# pl.title('Source')
# pl.subplot(1,2,2)
# pl.hist2d(xt[:,0],-xt[:,1],bins=n,range=[[-3,3],[-3,3]],cmap='Blues')
# pl.title('Target')
# pl.savefig('OT_dual_circle_new.png')


# Sinkhorn divergence (POT) rescaled entropy reg
M = ot.dist(xs, xt)+reg_val*np.log(ns*nt)
w_sinkhorn, log = ot.bregman.sinkhorn2(np.ones((ns,1))/ns, np.ones((nt,1))/nt, M, reg=reg_val, numItermax=10000, stopThr=1e-12, log=True)
plan_sinkhorn = np.reshape(log['u'], (-1,1))*np.exp(-M/reg_val)*np.reshape(log['v'], (1,-1))
u_sinkhorn = np.log(log['u'])*reg_val
v_sinkhorn = np.log(log['v'])*reg_val
u_sinkhorn_2 = (-np.log(log['u'])-0.5)*reg_val
v_sinkhorn_2 = (-np.log(log['v'])-0.5)*reg_val
d_sinkhorn = np.sum(u_sinkhorn)/ns + np.sum(v_sinkhorn)/nt - reg_val*np.sum(np.exp((u_sinkhorn + v_sinkhorn.T - M)/reg_val))


# Dual OT Stochastic Optimization
discreteOTComputer = PyTorchStochasticDiscreteTransport(xs, ws, xt, wt, 'entropy', reg_val)
history = discreteOTComputer.learn_OT_dual_variables(epochs=500, batch_size=5, lr=0.001)
d_stochastic = discreteOTComputer.compute_OT_MonteCarlo(epochs=20, batch_size=20)


pl.figure(2)
pl.plot(history['losses'], lw=3, label='loss')
pl.legend(loc='best')
pl.title('Loss per epoch')
pl.savefig('loss_per_epoch.png')


print('\n')
print('sinkhorn: %f' % d_sinkhorn)
print('stochastic dual: %f\n' % d_stochastic)

