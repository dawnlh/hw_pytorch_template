import numpy as np
import os
import hydra
import torch
import warnings
from omegaconf import OmegaConf
from importlib import import_module
from srcs.trainer._engine import train_engine
from importlib import import_module

# fix random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# ignore warning
warnings.filterwarnings('ignore')

# config: basenet_train


@hydra.main(config_path='conf', config_name='basenet_train')
def main(config):
    ## GPU setting
    if not config.gpus or config.gpus == -1:
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = config.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus)) # set visible gpu ids
    assert len(gpus) <= torch.cuda.device_count(), f'There are {torch.cuda.device_count()} GPUs on this machine, but you assigned $gpus={gpus}.'
    OmegaConf.set_struct(config, False) # enable access to non-existing keys
    config.n_gpus = len(gpus)

    ## show config
    # interpret hydra config and convert it to yaml
    config = OmegaConf.to_yaml(config, resolve=True)
    print('='*40+'\n', config, '\n'+'='*40+'\n') # print config info
    config = OmegaConf.create(config) # convert from yaml to OmegaConf

    ## training
    trainer_name = 'srcs.trainer.%s' % config.trainer_name
    training_module = import_module(trainer_name)
    train_engine(training_module.Trainer, gpus, config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
