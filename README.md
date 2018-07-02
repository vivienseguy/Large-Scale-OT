# Large-Scale-OT

Implementation in PyTorch of stochastic algorithms for the computation of Regularized Optimal Transport proposed in [1]

[1] Proposes:
 - A stochastic algorithm (Alg. 1) for computing the optimal dual variables of the regularized OT problem (from which the regularized-OT objective can be computed simply)
 - A stochastic algorithm (Alg. 2) for learning an Optimal Map, parameterized as a Deep Neural Network between the source and target probability measures
 
 Both entropy and L2 regularizations are considered and implemented.


### Requirements
```
python2 or python3
pytorch
matplotlib
```

### Usage
Start by creating the regularized-OT computation class: either PyTorchStochasticDiscreteOT or PyTorchStochasticSemiDiscreteOT depending on your setting. 
``` python
discreteOTComputer = PyTorchStochasticDiscreteOT(xs, ws, xt, wt, reg_type, reg_val, device_type=device_type, device_index=device_index)
```
Compute the optimal dual variables through Alg. 1.:
``` python
import numpy as np
from PyTorch.StochasticOTClasses.StochasticOTDiscrete import PyTorchStochasticDiscreteOT
history = discreteOTComputer.learn_OT_dual_variables(epochs=1000, batch_size=50, lr=0.0005)
```
Once the optimal dual variables have been obtained, you can compute the OT loss stochastically:
``` python
d_stochastic = discreteOTComputer.compute_OT_MonteCarlo(epochs=20, batch_size=50)
``` 
You can also learn an approximate optimal map between the two probability measures by learning the barycentric mapping (ALg. 2.). The mapping is parameterized as a deep neural network that you can supply in the functions parameters. Otherwise a default small 3-layers NN is used.
``` python
bp_history = discreteOTComputer.learn_barycentric_mapping(epochs=300, batch_size=50, lr=0.000002)
``` 
Once learned, you can apply the (approximate) optimal mapping to some sample via:
``` python
xsf = discreteOTComputer.evaluate_barycentric_mapping(xs)
``` 
You can visualize the source, target and mapped samples:
``` python
import matplotlib.pylab as pl
pl.figure()
pl.plot(xs[:, 0], xs[:, 1], '+b', label='source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='target samples')
pl.plot(xsf[:, 0], xsf[:, 1], '+g', label='mapped source samples')
pl.legend()
``` 
 
## References

[1] Seguy, Vivien and Damodaran, Bharath Bhushan and Flamary, Rémi and Courty, Nicolas and Rolet, Antoine and Blondel, Mathieu. [Large-Scale Optimal Transport and Mapping Estimation](https://arxiv.org/abs/1711.02283). Proceedings of the International Conference in Learning Representations (2018)



```bash
@inproceedings{seguy2018large,
  title={Large-Scale Optimal Transport and Mapping Estimation},
  author={Seguy, Vivien and Damodaran, Bharath Bhushan and Flamary, R{\'e}mi and Courty, Nicolas and Rolet, Antoine and Blondel, Mathieu},
  booktitle={Proceedings of the International Conference in Learning Representations},
  year={2018},
}
```







