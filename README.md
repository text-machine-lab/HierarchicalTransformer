# U-Net Transformer

[Paper for this repository can be found here](https://arxiv.org/abs/1910.10488)

This is a repository to add hierarchy to the Transformer. It is currently under development.

In addition to a fork from jadore801120 Transformer repository, also used baseline code from this repository:
https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling

Due to the combination of different repositories in one, the structure is a bit layered to say the least.

MIT licenses for each can be found in the licenses folder.

# Installation

Must run ubuntu_preprocess.py, cornell_preprocess.py and personachat_preprocess.py after downloading each of the required datasets and placing them in the data/ folder. This repository uses python3 with Pytorch.

# Run

The train.py module is the top-level directory can be used for U-Net/baseline training on a conversational Twitter corpus. In addition, models/train.py can be used for training on the Ubuntu Dialogue, Cornell Movie Dialogue, and PersonaChat datasets for perplexity evaluation. Wandb package used for visualization of loss curves and final results, and can be found here https://www.wandb.com/ and installed with

```pip install wandb```

In addition, configurations can be changed in the configs.py file. In general, inheriting from multiple repositories has created a convoluted structure within the repo.
