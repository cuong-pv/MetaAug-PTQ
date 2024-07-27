<!-- <div align="center">    -->
  
## This repository provides the implementation of our ECCV 2024 paper: [MetaAug: Meta-Data Augmentation for Post-Training Quantization](https://arxiv.org/abs/2407.14726)
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


## Training
### Download the warmup unet models from [here](https://drive.google.com/file/d/14lWmQ1oLo9jyH2O-yB9vSXluKYgMdUAb/view?usp=sharing) and put it in the `./warmup` folder.
### Postraining quantization on ImageNet
  ```bash
  bash run.sh
  ```

## License <a name="license"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.


## Citation <a name="citation"></a>

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@article{pham2024metaaug,
  title={MetaAug: Meta-Data Augmentation for Post-Training Quantization},
  author={Pham, Cuong and Dung, Hoang Anh and Nguyen, Cuong C and Le, Trung and Phung, Dinh and Carneiro, Gustavo and Do, Thanh-Toan},
  journal={arXiv preprint arXiv:2407.14726},
  year={2024}
}
```

## Related resources

This work based on resource below. A huge thank you to the original authors and community for their contributions to the open-source community.

- [Genie: Show Me the Data for Quantization](https://github.com/SamsungLabs/Genie/tree/main)


