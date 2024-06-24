Here, you'll find the server inference code and logs. 
To facilitate a smoother introduction to our training pipeline, we also have provided a Kaggle notebook. This notebook will help you quickly understand and get started with the inferencing process.

*Disclaimer* Despite using exact same random seed, slight performance mismatch may occur due to the usage of different GPUs (P100 vs. A100), different versions of CUDA, and PyTorch. To obtain 100% reproducible result, use the provided script [sketch.py] with exact hardware configuration and file version (e.g. PyTorch), provided in [./dependencies].
