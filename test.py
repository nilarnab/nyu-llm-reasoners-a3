import torch
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
