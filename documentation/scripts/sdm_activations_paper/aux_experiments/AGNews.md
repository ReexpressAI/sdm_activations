# SDM Activations: Auxiliary Experiments: AGNews


## Overview

We additionally examine the standard 4-class ($|\mathcal{Y}| = 4$) AGNews dataset (available at https://huggingface.co/datasets/fancyzhx/ag_news), which was suggested by a reviewer as a possible additional experiment to consider. This experiment demonstrates the behavior of the SDM estimator on a multi-class classification dataset with $|\mathcal{Y}| > 2$.

The Similarity ($q$) is label conditional (by definition of Eq. 4), and Distance ($d$) is calculated label-wise (via the minimum in Eq. 5) as a guard against outlier label-wise skew in the distribution of $d_{nearest}$. As such, a priori there is no reason to assume a reliability breakdown as $|\mathcal{Y}|$ increases. As a practical matter, since the SDM estimator is by design a class- and prediction-conditional estimator at a fixed $\alpha$ across all classes, if one (or more) class(es) has unusually poor accuracy, then Alg. 1 may fail to find a finite $q'_{min}$, at which point the user will know/need to choose a less conservative $\alpha$, or marginalize/merge classes, as applicable. That is a feature, not a bug, of the SDM estimator. (Existing estimators fall apart even in the binary setting, which is the most general setting in the sense that we can re-encode other cases as binary classification, up to marginalization over labels, so we focus on that in the main text.) 

It is straightforward to train and test additional datasets with the publicly available code in this repo. **Here we confirm the SDM activation and estimator behavior is not limited to binary classification via the main results on AGNews for Phi-3.5-mini-instruct (Phi3.5+SDM with the main $SDM_{HR}$ estimator).** First, we show the calibration behavior. Next, we examine the training dynamics. We close with an examination of the interpretability-by-exemplar properties.

## Task: AGNews (4-class classification)

Background: The standard AGNews task is a classification of short news article sentences (including titles) into four categories.

AGNews labels: World (class 0), Sports (class 1), Business (class 2), Sci/Tech (class 3)

AGNews dataset size: $|\mathcal{D_{tr}}|=60k$, $|\mathcal{D_{ca}}|=60k$, $|\mathcal{D_{te}}|=7600$

## Results

**Takeaway from the following tables: The SDM estimator is well-calibrated at $\alpha=0.95$ across the held-out test set.**

Higher is better, across all tables. (These are the same columns as in the main text, with "Admitted proportion=$\frac{n}{|D|}$", but we split by conditioning to be legible in markdown formatting.)

### **Class-conditional**

| Dataset | Model | Estimator | y=0 Acc. | y=0 Admitted Proportion | y=1 Acc. | y=1 Admitted Proportion | y=2 Acc. | y=2 Admitted Proportion | y=3 Acc. | y=3 Admitted Proportion |
|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| AGNews Calibration | Phi3.5+SDM | No-Reject | 0.93 | 0.25 | 0.98 | 0.25 | 0.87 | 0.25 | 0.90 | 0.25 |
| AGNews Calibration | Phi3.5+SDM | SDM_HR | 0.99 | 0.16 | 1.00 | 0.22 | 0.98 | 0.13 | 0.98 | 0.08|
| AGNews test | Phi3.5+SDM | No-Reject | 0.92 | 0.25 | 0.98 | 0.25 | 0.85 | 0.25 | 0.89 | 0.25 |
| AGNews test | Phi3.5+SDM | SDM_HR | 0.98 | 0.15 | 1.00 | 0.22 | 0.97 | 0.13 | 0.98 | 0.08 |


### **Prediction-conditional**

| Dataset | Model | Estimator | $\hat{y}=0$ Acc. | $\hat{y}=0$ Admitted Proportion | $\hat{y}=1$ Acc. | $\hat{y}=1$ Admitted Proportion | $\hat{y}=2$ Acc. | $\hat{y}=2$ Admitted Proportion | $\hat{y}=3$ Acc. | $\hat{y}=3$ Admitted Proportion |
|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| AGNews Calibration | Phi3.5+SDM | No-Reject | 0.92 | 0.25 | 0.98 | 0.25 | 0.90 | 0.24 | 0.88 | 0.26 |
| AGNews Calibration | Phi3.5+SDM | SDM_HR | 0.99 | 0.16 | 1.00 | 0.22 | 0.98 | 0.13 | 0.97 | 0.09 | 
| AGNews test | Phi3.5+SDM | No-Reject | 0.91 | 0.25 | 0.98 | 0.25 | 0.89 | 0.24 | 0.87 | 0.26 |
| AGNews test | Phi3.5+SDM | SDM_HR | 0.99 | 0.15 | 1.00 | 0.22 | 0.98 | 0.13 | 0.97 | 0.08 |


### **Marginal**

| Dataset | Model | Estimator | $y \in \{0,1,2,3\}$ Acc. | $y \in \{0,1,2,3\}$ Admitted Proportion |
|-------------------|-------------------|-------------------|-------------------|-------------------|
| AGNews Calibration | Phi3.5+SDM | No-Reject | 0.92 | 1. |
| AGNews Calibration | Phi3.5+SDM | SDM_HR | 0.99 | 0.59 |
| AGNews test | Phi3.5+SDM | No-Reject | 0.91 | 1. |
| AGNews test | Phi3.5+SDM | SDM_HR | 0.99 | 0.58 |


## Training dynamics

We train for J=10 iterations each of 500 epochs, which is an increase from 200 epochs in the binary datasets examined in the main text given the much larger dataset. 

### Training dynamics: Iteration 5 (the chosen model by lowest Balanced SDM loss over $\mathcal{D_{ca}}$)

**After epoch 1 (i.e., after standard cross-entropy, $q=e-2, d=1$)**

Training set Balanced SDM Loss: 0.392

Training set Balanced Accuracy: 0.889

Calibration (Ca) set Balanced SDM loss: 0.389

Ca Balanced mean q: 103.1

Ca Class 0, $d_{nearest}$: Median: 6.3, Max: 44.7

Ca Class 1, $d_{nearest}$: Median: 3.5, Max: 53.4

Ca Class 2, $d_{nearest}$: Median: 3.9, Max: 41.3

Ca Class 3, $d_{nearest}$: Median: 5.0, Max: 48.8

Ca Balanced Accuracy: 0.891

**After epoch 200:**

Training set Balanced SDM Loss: 0.211

Training set Balanced Accuracy: 0.919

Calibration (Ca) set Balanced SDM loss: 0.234

Ca Balanced mean q: 327.8

Ca Class 0, $d_{nearest}$: Median: 13.4, Max: 296.3

Ca Class 1, $d_{nearest}$: Median: 5.7, Max: 277.0

Ca Class 2, $d_{nearest}$: Median: 7.6, Max: 206.1

Ca Class 3, $d_{nearest}$: Median: 9.7, Max: 226.1

Ca Balanced Accuracy: 0.912

**After epoch 492 (chosen epoch)**:

Training set Balanced SDM Loss: 0.178

Training set Balanced Accuracy: 0.929

Calibration (Ca) set Balanced SDM loss: 0.218

Ca Balanced mean q: 433.2

Ca Class 0, $d_{nearest}$: Median: 14.9, Max: 396.0

Ca Class 1, $d_{nearest}$: Median: 6.3, Max: 275.5

Ca Class 2, $d_{nearest}$: Median: 8.1, Max: 345.0

Ca Class 3, $d_{nearest}$: Median: 10.5, Max: 253.5

Ca Balanced Accuracy: 0.919

**Takeaway: The general training dynamics trend is that $q$ is increasing, and outlier $d_{nearest}$ values, as applicable, are pushed away as training proceeds, while the loss decreases and accuracy trends upwards. I.e., the functional form of Eq. 7 is behaving as desired on this non-binary classification task.**

## Instance-wise Interpretability-by-exemplar

We also examine the interpretability-by-exemplar behavior on the AGNews dataset. 

### Example 1 (from test set): 

Highest probability example in the HR region, but $y\neq\hat{y}$:

> ```Argentina Beats U.S. Men's Basketball Team Argentina defeated the United States team of National Basketball Association stars 89-81 here Friday in the Olympic semi-finals, dethroning the three-time defending champions.```

> Ground-truth label: $y$=World (0)

SDM estimator output:

$\hat{y}$=Sports (1)

$sdm(z')\approx [0.0, 1.0, 0.0, 0.0]$

$d\approx0.858$

$q'=2048$

$d_{lower} \approx 0.843$, $d_{upper} \approx 0.873$ (See Appendix A.6)

$sdm(z')_{lower} \approx [0.0, 1.0, 0.0, 0.0]$ (See Appendix A.6)

Nearest training match:

> ```Puerto Rico Stuns U.S. in Opening Round Puerto Rico upsets the United States, 92-73, at the men's basketball preliminaries on Sunday, the first loss at the Games for the three-time defending gold medalists since 1988.```

> Ground-truth label: $y$=Sports (1)

**This is a case of ground-truth label error/ambiguity (i.e., aleatoric/irreducible error) in the test set.**

### Example 2 (from test set): 

Highest probability example in the HR region, $y = \hat{y}$:

> ```Drew Out of Braves' Lineup After Injury (AP) AP - Outfielder J.D. Drew missed the Atlanta Braves' game against the St. Louis Cardinals on Sunday night with a sore right quadriceps.```

> Ground-truth label: $y$=Sports (1)

SDM estimator output:

$\hat{y}$=Sports (1)

$sdm(z')\approx [0.0, 1.0, 0.0, 0.0]$

$d \approx 0.994$

$q'=2048$

$d_{lower} \approx 0.979$, $d_{upper} \approx 1.0$ (See Appendix A.6)

$sdm(z')_{lower} \approx [0.0, 1.0, 0.0, 0.0]$ (See Appendix A.6)

Nearest training match:

> ```Cards Second Baseman Tony Womack Injured (AP) AP - St. Louis Cardinals second baseman Tony Womack left Monday night's game against Houston in the seventh inning after getting hit on the left hand with a pitch.```

> Ground-truth label: $y$=Sports (1)


**Takeaway: The interpretability-by-exemplar properties are present on this non-binary classification task.**

## Conclusion

The SDM activation and estimator are not limited to binary classification.

## Appendix: Scripts

The data preprocessing scripts and training/eval scripts are available in this repo in `agnews_phi3.5_data.sh` and `sdm_estimator_phi_agnews_0.95.sh`, respectively.

## Appendix: Prompt used

> ```Here is a news article. <article> DOCUMENT </article> Classify the article into one of the following four categories: World (0), Sports (1), Business (2), Sci/Tech (3). Only respond with the category number.```

In this case, we generate for two steps, since the Phi-3.5-mini-instruct tokenizer prepends the separate special-symbol underscore to the numbers (0, 1, 2, 3). See `add_phi_3_5_instruct_embeddings_agnews.py`.

## Appendix: Summary Stats determining HR region

$\psi \approx [0.9796, 1.0000, 0.9744, 0.9505]$

$q'_{min} \approx 22.7$

## Appendix: Compute

The 5,000 training epochs (after caching $h$) took approximately 24 hours on an Nvidia L4 GPU. (That is for all J=10 iterations, each of which consists of 500 epochs.) The batched forward adaptor pass, including dense matching and all SDM calculations (using cached $h$; i.e., excluding the forward pass over Phi-3.5-mini-instruct to determine $h$), over **all 7.6k test instances took approximately 7 SECONDS** (not a typo) on an Nvidia L4 GPU. Using optimized L2 matching with established packages like FAISS for GPUs and vectorizing the calculations of $q$ and CDF lookup results in quite reasonable inference time with a $|\mathcal{D_{tr}}|=60k$. 

For perspective, the Phi-3.5-mini-instruct forward pass to cache $h$ over all 7.6k AGNews test instances took approximately 1101.7 seconds. That is with a batch size of 1 to simplify controlling for numerical differences with batched attention masks with these models, but assuming the ideal setting of perfectly efficient batching, that would still be approximately 22 seconds for each batch of 50 as used for the SDM calculations, compared to the real wall clock time of 7 seconds for the SDM calculations for ALL 7.6k instances. **That is, at this scale, compute is overwhelmingly dominated by the forward pass of the underlying LM.**
