from configs.cfg_loader import load_config
cfg = load_config("resnet18")

print(cfg.BATCH_SIZE)
