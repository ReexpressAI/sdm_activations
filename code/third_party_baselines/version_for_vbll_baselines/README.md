# VBLL Baselines

## Overview

This directory contains the version of the code base used to run the variational Bayesian last-layer neural networks (https://github.com/VectorInstitute/vbll) baselines. The additional files have the letters "vbll" in the filename. There are also a handful of additional command line options in reexpress.py. Consult the run scripts in `documentation/scripts/sdm_activations_paper/models/additional_baselines_vbll` to replicate the experiments. We include this for provenance of the baseline experiments, but for normal use of SDM activations and estimators we recommend using the main code base in the [Reexpress MCP Server repo](https://github.com/ReexpressAI/reexpress_mcp_server), as the code in this directory will likely not receive further updates.

## Setup

The VBLL baselines were run by cloning the main enivornment and installing the VBLL package:

```
conda create --name re_mcp_v200_vbll_comparison --clone re_mcp_v200

conda activate re_mcp_v200_vbll_comparison

pip install vbll

# Successfully installed vbll-0.4.9
```

All runs used a single A100 GPU, but a consumer GPU or just using CPU or MPS devices should be sufficient.
