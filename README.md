## **[MDBN](https://github.com/thy960112/MDBN)**

![Architecture](https://github.com/thy960112/MDBN/blob/main/architecture.png)

**[Multi-Depth Branches Network for Efficient Image Super-Resolution](https://arxiv.org/abs/2309.17334)**

Huiyuan Tian, Li Zhang, Shijian Li, Min Yao and Gang Pan

### Run

#### Environments

1. Install python3.10
2. Install PyTorch (tested on Release 1.12)
3. [BasicSR 1.4.2](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md)

#### Installation
```
# Clone the repo
git clone https://github.com/thy960112/MDBN.git
# Install dependent packages
cd MDBN
pip install -r requirements.txt
# Install BasicSR
python setup.py develop
```

#### Training

For different scales, the following commands can be used for training respectively:

```
# train MDBN for x2 SR
python basicsr/train.py -opt options/train/MDBN/train_MDBN_x2.yml
# train MDBN for x3 SR
python basicsr/train.py -opt options/train/MDBN/train_MDBN_x3.yml
# train MDBN for x4 SR
python basicsr/train.py -opt options/train/MDBN/train_MDBN_x4.yml
```

#### Testing

1. Download the [pre-trained models](https://github.com/thy960112/MDBN/tree/main/experiments/pretrained_models).
2. Download the testing dataset.
3. For different scales, the following commands can be used for testing respectively:

```
# test MDBN for x2 SR
python basicsr/test.py -opt options/test/MDBN/test_MDBN_x2.yml
# test MDBN for x3 SR
python basicsr/test.py -opt options/test/MDBN/test_MDBN_x3.yml
# test MDBN for x4 SR
python basicsr/test.py -opt options/test/MDBN/test_MDBN_x4.yml
```

#### Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.
