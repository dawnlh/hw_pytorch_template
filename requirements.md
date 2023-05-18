## conda env configuration
> Note: multi-package can be installed simutaneously with the following command
> `conda install pkg==verx.x. pkg2==verx.x.`
> `pip install pkg==verx.x. pkg2==verx.x.`

### conda package
- torch, torchvision # refer to https://pytorch.org/get-started/locally/, e.g. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
- opencv # `conda install opencv`
- scikit-image # `conda install scikit-image`
- matplotlib # `conda install matplotlib`
- tensorboard>=1.14 # `conda install tensorboard`
- pandas # `conda install pandas`
- tqdmp # `conda install tqdmp`

###  pip package
    hydra-core>=1.1
    hydra_colorlog

    kornia
    einops
    pytorch-gradual-warmup-lr
    pytorch-msssim
    ptflops
    pyiqa
    requests
