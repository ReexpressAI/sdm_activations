# SDM Activations: Auxiliary Experiments

## Overview

This directory includes additional experiments to further demonstrate aspects of the behavior of SDM activations and estimators. These are not needed to understand the main paper, but they provide additional views of the overall behavior and motivation of the modeling choices.

## AGNews (4-class classification)

[AGNews.md](AGNews.md) demonstrates that the behavior of the SDM estimator is not restricted to binary classification. This is shown by examining the training dynamics, calibration results, and interpretability-by-exemplar behavior on the standard AGNews (4-class classification) dataset.

## Adaptor Representation Ablation

[AdaptorRepresentationAblation.md](AdaptorRepresentationAblation.md) demonstrates that Alg. 1 is robust to representations held constant during the optimization of Eq. 7. This is shown by directly using the hidden-states of the underlying model as the representations used for matching instead of those of a learned CNN. 

## Appendix: Ensembles

Although not considered in the main text, it is straightforward to ensemble all J models with the existing training code in this repo. During training, the model files for each iteration are saved in a directory identified by the iteration number ('0/', '1/', etc.). Ensembling can be achieved by running eval with the model from each iteration[^1], and then merging/filtering the output predictions files. In very high-risk settings, one could require, for example, that a point only be admitted if the predictions across all J iterations match AND the point falls into the High-Reliability region for all J iterations. We leave a systematic examination of such ensembling behavior to future work.

[^1]: To run inference with the models in the iteration directories using the existing test script, you need to first simply copy the `global_uncertainty_statistics.json` file from the main directory into the applicable iteration directory.
