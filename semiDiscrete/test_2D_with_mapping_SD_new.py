#%% 2-dimensional test script
import matplotlib.pylab as pl
import numpy as np
from keras.models import Input, Model, Sequential
from keras.layers import Dense
import SGD_OT_SD


def get_sample_network(d, name=None):
    u=Sequential(name=name)
    u.add(Dense(256, input_dim=d,activation='relu'))
    u.add(Dense(256, input_dim=d, activation='relu'))
    u.add(Dense(1, activation='linear'))
    return u


d=2
nt=200
scale = 2.5
xt = np.concatenate((np.random.choice(np.array([-scale, scale]), nt).reshape((-1,1)), np.random.choice(np.array([-scale, scale]), nt).reshape((-1,1))),1)
xt = xt+0.2*np.random.randn(nt,2)


# Dual OT Stochastic Optimization
myTransporterDual = SGD_OT_SD.StochasticSemiDiscreteTransport(reg_type='entropy', reg_val=1.,  model_potential_fun=get_sample_network, xt=xt, wt=np.ones((nt,))/nt)
h, _= myTransporterDual.fit_from_gaussian(lr=0.1, epochs=10000, batch_size=100, mean=0., std=0.5, processor_type='cpu')

sav_results={}
res={}

res['dual']={}
res['dual']['loss']=h['losses']
res['dual']['time']=h['time']
res['dual']['epochs']=10

pl.figure(1)
pl.plot(h['losses'],lw=3,label='loss')
pl.title('Learning')
pl.savefig('OT_dual_loss.png')


# Stochastic Barycentric Mapping Learning
h = myTransporterDual.fit_barycentric_mapping_from_gaussian(lr=0.000001, epochs=2000, batch_size=100, mean=0., std=0.5, processor_type='cpu')

pl.figure(2)
pl.plot(h['losses'],lw=3,label='loss')
pl.title('Learning')
pl.savefig('Barycentric_map_learning_loss.png')


# Compute Mapping estimation
idv = np.random.permutation(nt)
xs = np.random.normal(size=(nt,d))
xsf = myTransporterDual.predict_f(xs)

pl.figure(3)
pl.plot(xs[:, 0], xs[:, 1], '+b')
pl.plot(xt[:, 0], xt[:, 1], 'xr')
pl.plot(xsf[:, 0], xsf[:, 1], '+g')
pl.savefig('mapping_estimation.png')

