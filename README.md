# Com-BrainTF
The official Pytorch implementation of paper "Community-Aware Transformer for Autism Prediction in fMRI Connectome" accepted by **MICCAI 2023**.

Abide dataset available [here](https://drive.google.com/file/d/1rTmBuLbMNu-vW7g43eSu21ur1Sc4oVHh/view?usp=sharing).

1. Update *path* in the file *source/conf/dataset/ABIDE.yaml* to the path of your dataset.

2. *node_clus_map.pickle* contains the ROI-Functional Network assignments (a dictionary of the form ROI:Community {0:1, 1:3, .... 199:7}). 
For ABIDE, ROI-Functional Network assignments follows the Yeo 7 network template (Yeo et al. J Neurophysiol. 2011). Replace this file if you use a different dataset.

3. Run the following command to train the model.

```bash
python -m source --multirun datasz=100p model=comtf dataset=ABIDE repeat_time=5 preprocess=non_mixup
```
- **datasz**, default=(10p, 20p, 30p, 40p, 50p, 60p, 70p, 80p, 90p, 100p). Percentage of the total number of samples in the dataset to use for training.

- **model**, default=(comtf,fbnetgen,brainnetcnn). Model to be used.

- **dataset**, default=(ABIDE). Dataset to be used.

- **repeat_time**, default=5. Number of times to repeat the experiment.

- **preprocess**, default=(mixup, non_mixup). Data pre-processing.

## Dependencies

  - python=3.9
  - cudatoolkit=11.3
  - torchvision=0.13.1
  - pytorch=1.12.1
  - torchaudio=0.12.1
  - wandb=0.13.1
  - scikit-learn=1.1.1
  - pandas=1.4.3
  - hydra-core=1.2.0

## Acknowledgement

We built Com-BrainTF on top of [BNT](https://github.com/Wayfear/BrainNetworkTransformer/tree/main)
