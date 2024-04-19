import torch
import os
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

print(os.environ['CUDA_HOME'])
print(os.environ['PATH'])
