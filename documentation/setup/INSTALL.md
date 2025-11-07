
# Create the Conda environment for running on Linux with CUDA GPUs (currently tested with A100 GPUs)

```bash
conda create -n re_mcp_v200 python=3.12
conda activate re_mcp_v200
pip install torch==2.7.1 transformers==4.53.0 accelerate==1.8.1 numpy==1.26.4
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.12.0
conda install -c conda-forge matplotlib=3.10.0
pip install datasets==4.0.0
pip install tqdm==4.67.1
pip install tensorboard==2.20.0
```

