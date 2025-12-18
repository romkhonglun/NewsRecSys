import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
if __name__ == "__main__":
    print("This is the main module.")