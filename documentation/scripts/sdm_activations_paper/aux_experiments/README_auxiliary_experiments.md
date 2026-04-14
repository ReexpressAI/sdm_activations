# SDM Activations: Auxiliary Experiments

## Overview

This directory includes additional experiments to further demonstrate aspects of the behavior of SDM activations and estimators. These are not needed to understand the main paper, but they provide additional views of the overall behavior and motivation of the modeling choices.

## AGNews (4-class classification)

[AGNews.md](AGNews.md) demonstrates that the behavior of the SDM estimator is not restricted to binary classification. This is shown by examining the training dynamics, calibration results, and interpretability-by-exemplar behavior on the standard AGNews (4-class classification) dataset.

## Adaptor Representation Ablation

[AdaptorRepresentationAblation.md](AdaptorRepresentationAblation.md) demonstrates that Alg. 1 is robust to representations held constant during the optimization of Eq. 7. This is shown by directly using the hidden-states of the underlying model as the representations used for matching instead of those of a learned CNN. 

## Appendix: Ensembles

Although not considered in the main text, it is straightforward to ensemble all J models. During training, the model files for each iteration are saved in a directory identified by the iteration number ('0/', '1/', etc.). Ensembling can be achieved by running eval with the --eval_ensemble flag in release v2.3.0 of the code. (See the scripts in this repo for examples.) This constructs a High-Reliability region by only admitting a point if the predictions across all J iterations match AND the point falls into the High-Reliability region for all J iterations. For reference, this is done for HR, as well as HR_lower, constructed using d_lower (Eq. 14). (For the points outside the HR/HR_lower region, we take as the prediction the most frequent prediction across the J ensembles, with ties randomly broken. The associated uncertainty estimate is simply taken as the lowest probability among the J estimates whose argmax prediction matches that of the chosen ensemble prediction.) In this way, we can directly control for noise across shuffles of the data and parameter initializations. As demonstrated in the main text, endogenous modeling of Similarity and Distance is already a step change from estimators predicated only on the output logits, but this additional source of uncertainty is worth considering in high-risk settings. We leave a systematic examination of such ensembling behavior to future work.
