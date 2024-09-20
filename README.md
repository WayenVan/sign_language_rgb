# csi_sign_language

This repo is for CTC-based Continuous Sign Langauge Recogntion. Powered by [pytorch_lightning](https://lightning.ai/docs/pytorch/stable/) and [hydra](https://github.com/facebookresearch/hydra)

# requirements

```
mmcv                      2.1.0
mmdet                     3.2.0
mmengine                  0.10.4
mmflow                    0.5.2
mmpose                    1.3.2
mmpretrain                1.2.0
click                     8.1.7
pytorch-lightning         2.3.3
hydra-core                1.3.2

```

# config files

check all config files in `configs/run/*.yaml`

# run training

```bash
python3 ./tools/train_lightning.py --config-name <selected_config_name>

#forexampe
python3 ./tools/train_lightning.py --conig-name run/train/resnet_trans_lightning
```

````
