<!-- <div align="center">    -->
  
# This repository provides the implementation of our ECCV 2024 paper: [MetaAug: Meta-Data Augmentation for Post-Training Quantization
](https://arxiv.org/abs/2407.14726)
</div>


## Installation

Prerequires:

- Python 3.6
- PyTorch 1.9.0
- torchvision 0.10.0
- other packages like wandb, tensorboard, etc.

Install the package:

```
pip3 install -r requirements.txt
```

## Dataset
For ImageNet dataset, download data to train and val folder

## Training

### Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`
  ```bash
  bash run.sh
  ```

## License <a name="license"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Related resources

This work based on resource below. A huge thank you to the original authors and community for their contributions to the open-source community.

- [Genie: Show Me the Data for Quantization](https://github.com/SamsungLabs/Genie/tree/main)

