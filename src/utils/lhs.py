import torch

def latin_hypercube_sampling_1D(N):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intervals = torch.linspace(-1, 1, N, dtype=torch.float32, device=device)
    points = intervals + torch.rand(N, dtype=torch.float32, device=device) / N
    shuffled_indices = torch.randperm(N, device=device)
    points_shuffled = points[shuffled_indices]
    
    return points_shuffled