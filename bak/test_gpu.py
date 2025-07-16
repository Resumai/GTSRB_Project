import torch
import time

# Dummy matrix multiplication test
# Sub-second results woul show that cuda acceleration and gpu works properly
x = torch.randn(10000, 10000, device="cuda")
y = torch.randn(10000, 10000, device="cuda")

start = time.time()
z = x @ y
torch.cuda.synchronize()  # ensure all CUDA ops finish
print("Matrix multiply took", time.time() - start, "seconds")


# For checking if GPU is detected
# print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))