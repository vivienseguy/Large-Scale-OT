# Large-Scale-OT

Implementation in PyTorch of stochastic algorithms for the computation of Regularized Optimal Transport proposed in [1]

[1] Proposes:
 - A stochastic algorithm (Alg. 1) for computing the optimal dual variables of the regularized OT problem (from which the regularized-OT objective can be computed simply)
 - A stochastic algorithm (Alg. 2) for learning an Optimal Map, parameterized as a Deep Neural Network between the source and target probability measures
 
 Both entropy and L2 regularizations are considered and implemented.


### Requirements
```
python3
pytorch
matplotlib
...

### Usage

 
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







