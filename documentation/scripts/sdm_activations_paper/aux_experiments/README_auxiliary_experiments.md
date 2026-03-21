# SDM Activations: Auxiliary Experiments

## Overview

This directory includes additional experiments to further demonstrate aspects of the behavior of SDM activations and estimators. These are not needed to understand the main paper, but they provide additional views of the overall behavior and motivation of the modeling choices.

## AGNews (4-class classification)

[AGNews.md](AGNews.md) demonstrates that the behavior of the SDM estimator is not restricted to binary classification. This is shown by examining the training dynamics, calibration results, and interpretability-by-exemplar behavior on the standard AGNews (4-class classification) dataset.

## Adaptor Representation Ablation

[AdaptorRepresentationAblation.md](AdaptorRepresentationAblation.md) demonstrates that Alg. 1 is robust to representations held constant during the optimization of Eq. 7. This is shown by directly using the hidden-states of the underlying model as the representations used for matching instead of those of a learned CNN. 
