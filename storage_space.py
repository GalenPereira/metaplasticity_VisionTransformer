import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(device))
    
    # Allocate some tensors on the GPU
    tensor1 = torch.randn(1000, 1000).to(device)  # Allocate a random tensor
    tensor2 = torch.zeros(500, 500).to(device)    # Allocate a zero tensor
    
    # Print the allocated memory after allocating tensors
    allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 3
    print("Allocated memory:", allocated_memory, "GB")
    
    # Print the maximum allocated memory after allocating tensors
    max_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print("Max memory:", max_memory, "GB")
    
    # Print the total memory available on the GPU
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    print("Total memory:", total_memory, "GB")
    
else:
    print("CUDA is not available. You are using CPU.")
