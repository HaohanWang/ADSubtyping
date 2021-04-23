import os

def prepare_train_directories(config):
  out_dir = config.train.dir
  os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)