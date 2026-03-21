# SDM Activations: Auxiliary Experiments: Adaptor Representation Ablation


## Overview

The SDM estimator is intended to be robust to the optimization process that determines the parameters of the adaptor of the SDM activation. Alg. 1 runs after training is completed, as noted in the main text. In this experiment, we examine an extreme version of this by directly using the hidden-states of the underlying model as the representations used for matching, instead of those of a learned CNN.

## Ablation of the Adaptor representations

The CNN adaptor, $g$, is a distillation of the network. It is a very simple structure; combined with the final linear layer there is no non-linearity within that structure. Compare that with the MLP of the VBLL estimator, which has an inner non-linearity. The goal with $g$ is to keep it as simple as possible while still having the following properties, one or more of which are typically desired in real applications: A reduction in the dimension of $h$ (for reducing storage and retrieval costs); improved statistical efficiency (i.e., fewer rejected points, ceteris paribus, by optimizing against the desired reference class), which is obviously task/data dependent, but for which we seek to at least do no worse than directly using the hidden-states; ability to compose with attributes or other networks when additional capacity is needed (not examined in this work); straightforward conversion to hard-attention via maxpool, if needed for interpretability at lower resolutions, e.g., word-level (not examined in this work).

The SDM estimator is itself relatively robust to the output of $g$, in general, since Alg. 1 runs after training is complete. We empirically examine this here to provide additional insight on the training dynamics of the SDM activation.

Holding everything else constant, we replace $g$ with the identity function. Thus, $h'=h \in R^{6144}$ for Phi3.5 (`Phi3.5+g(identity)+SDM`) and $h'=h \in R^{8192}$ for Mixtral (`Mixtral8x7b+g(identity)+SDM`). This is still a non-trivial adaptor, because it retains the final linear layer, but $h'$ is held fixed.

**This tests whether Alg. 1 is robust to representations held constant during the optimization of Eq. 7.**

### Results (**Marginal**)

| Dataset | Model | Estimator | $y \in \{0,1\}$ Acc. | $y \in \{0,1\}$ Admitted Proportion |
|-------------------|-------------------|-------------------|-------------------|-------------------|
| Factcheck Calibration | Phi3.5+g(identity)+SDM | SDM_HR | 1.0 | 0.26 |
| Factcheck | Phi3.5+g(identity)+SDM | SDM_HR | 1.0 | 0.11 |
| Sentiment Calibration | Phi3.5+g(identity)+SDM | SDM_HR | 0.99 | 0.51 |
| Sentiment | Phi3.5+g(identity)+SDM | SDM_HR | 1.0 | 0.49 |
| Sentiment OOD | Phi3.5+g(identity)+SDM | SDM_HR | R | 0.0 |
| Factcheck Calibration | Mixtral8x7b+g(identity)+SDM | SDM_HR | 1.0 | 0.1 |
| Factcheck | Mixtral8x7b+g(identity)+SDM | SDM_HR | 1.0 | 0.04 |
| Sentiment Calibration | Mixtral8x7b+g(identity)+SDM | SDM_HR | 0.99 | 0.60 |
| Sentiment | Mixtral8x7b+g(identity)+SDM | SDM_HR | 1.0 | 0.61 |
| Sentiment OOD | Mixtral8x7b+g(identity)+SDM | SDM_HR | 1.0 | 0.0006 |


The estimator remains well-calibrated at $\alpha$ (conditional and other splits omitted for space, but the pattern holds). That is ideal: We want Alg. 1 to be robust to the optimization of Eq. 7. But other than for the dimension reduction, does learning $g$ matter? For these models/tasks, in most cases, yes: Learning $g$ in most of these cases increases the proportion of points in the HR region. Note the substantive reduction in the `Admitted Proportion` when $g$ is identity compared to the results in the paper. The in-distribution behavior is particularly telling of what is going on here in terms of improved sample efficiency of the HR region when learning $g$. For example, in Table 4 in the paper, 69% of calibration points for the Sentiment dataset are admitted for `Phi3.5+SDM`, which drops to 51% for `Phi3.5+g(identity)+SDM` (and from 74% to 60% for Mixtral). The optimization is more constrained by holding $g$ at identity. The one exception is the Factcheck dataset for Mixtral8x7b, where `Mixtral8x7b+g(identity)+SDM` is able to partition a region at $\alpha=0.95$, but the size of that region is relatively small (only 10% of in-distribution points).

### Ex. Training dynamics for `Mixtral8x7b+g(identity)+SDM`: Iteration 6 (the chosen model by lowest Balanced SDM loss over $\mathcal{D_{ca}}$):

To examine this further, as an illustrative example, we include the behavior during training below for `Mixtral8x7b+g(identity)+SDM` over the Sentiment dataset.

**After epoch 1 (i.e., after standard cross-entropy, $q=e-2, d=1$)**

Training set Balanced SDM Loss: 0.122

Training set Balanced Accuracy: 0.945

Calibration (Ca) set Balanced SDM loss: 0.120

Ca Balanced mean q: 197.1

Ca Class 0, $d_{nearest}$: Median: 406.7, Max: 1712.3

Ca Class 1, $d_{nearest}$: Median: 494.3, Max: 2360.3

Ca Balanced Accuracy: 0.943

**After epoch 195 (chosen epoch)**:

Training set Balanced SDM Loss: 0.092

Training set Balanced Accuracy: 0.961

Calibration (Ca) set Balanced SDM loss: 0.099

Ca Balanced mean q: 197.1

Ca Class 0, $d_{nearest}$: Median: 406.3, Max: 1595.5

Ca Class 1, $d_{nearest}$: Median: 497.1, Max: 2360.3

Ca Balanced Accuracy: 0.953

**Takeaway: The general training dynamics trend is that $q$ and $d_{nearest}$ do not substantively change, which is to be expected since $g$ is held fixed at identity, while parameter changes to the linear layer do at least allow the loss to decrease and accuracy to trend upwards.**

## Conclusion

Alg. 1 is robust to representations held constant during the optimization of Eq. 7.

## Appendix: Replication

This is straightforward to replicate with a simple change to the code.

In `sdm_model.py`:

Replace the line

`self.fc = nn.Linear(self.exemplar_vector_dimension, self.numberOfClasses)`

with 

`self.fc = nn.Linear(exemplar_network_input_size, self.numberOfClasses)`

Next, replace the line

`batch_exemplar_vectors = self.conv(batch_exemplar_vectors).squeeze(2)`

with 

`batch_exemplar_vectors = batch_exemplar_vectors.squeeze(1)`

This has the effect of simply passing the input hidden-states directly to the linear layer. The self.conv structure will now be extraneous and unused (with randomly initialized weights), but for the purposes here, just keep it to avoid modifying the model saving and loading code.

The training and eval scripts provided for replicating the results in the main paper can then be used directly with the modified code.
