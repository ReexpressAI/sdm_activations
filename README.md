# SDM Activations and SDM Language Models research repo


## Overview

This repo includes support code and replication scripts for the papers "Similarity-Distance-Magnitude Activations" and "Similarity-Distance-Magnitude Language Models". This repo only includes auxiliary code (e.g., for preprocessing the research datasets) and scripts containing the parameters used for the experiments. The main code is in the [Reexpress MCP Server repo](https://github.com/ReexpressAI/reexpress_mcp_server), version 2.0.0 (Commit 78c8465). The preprocessed data is available in the GitHub release binaries in *this* repo.

## Installation

Create the conda environment in [INSTALL.md](documentation/setup/INSTALL.md). In our provided scripts, we assume Linux and CUDA GPUs, but the scripts should also work on cpu, or on Apple silicon ('mps'), if you install an applicable version of FAISS and adjust the command line options for the device, accordingly. 

## Experiments

### "Similarity-Distance-Magnitude Activations"

Scripts for training and testing the models are in the [sdm_activations_paper directory](documentation/scripts/sdm_activations_paper/models).

### "Similarity-Distance-Magnitude Language Models"

Scripts for training and testing the models are in the [sdm_lms_paper directory](documentation/scripts/sdm_lms_paper/models).

## Citations

```
@misc{Schmaltz-2025-SimilarityDistanceMagnitudeLanguageModels,
      title={Similarity-Distance-Magnitude Language Models}, 
      author={Allen Schmaltz},
      year={2025},
      eprint={2510.26183},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.26183}, 
}
```


```
@misc{Schmaltz-2025-SimilarityDistanceMagnitudeActivations,
      title={Similarity-Distance-Magnitude Activations}, 
      author={Allen Schmaltz},
      year={2025},
      eprint={2509.12760},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.12760}, 
}
```
