# run multi tasks sequentially
python train.py -m optimizer.lr=0.001,0.002

# resume training (change ckp, gpu, run_dir in the config file first)
python train.py --config-path exp_dir/.hydra --config-name config hydra.run.dir=exp_dir