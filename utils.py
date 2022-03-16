import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def base_kernel(x, y, sigma):
    norm_square = np.linalg.norm(x-y) ** 2
    sigma_square = sigma ** 2
    
    return np.exp(- norm_square /(2* sigma_square))

def composite_kernel(x, y, sigmas):
    result = 0
    
    for sigma in sigmas:
        result += base_kernel(x, y, sigma)
        
    return result

def compute_mmd(dataset_x, dataset_y, sigmas=[1, 5, 10, 15, 20]):
    result = 0
    n = len(dataset_x)
    m = len(dataset_y)
    
    for i in range(n):
        for j in range(n):
            result += 1./(n**2) * composite_kernel(dataset_x[i], dataset_x[j], sigmas)
    
    for i in range(n):
        for j in range(m):
            result -= 2./(n*m) * composite_kernel(dataset_x[i], dataset_y[j], sigmas)
    
    for i in range(m):
        for j in range(m):
            result += 1./(m**2) * composite_kernel(dataset_y[i], dataset_y[j], sigmas)
            
    return np.sqrt(result)
