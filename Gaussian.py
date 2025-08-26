import numpy as np
import math

def getKernal(sigma=1, radius=1):
    # sigma = radius / 2
    kernalSize = radius * 2 + 1
    
    # Create Kernal
    kernal = np.zeros((kernalSize, kernalSize))
    k_sum = 0
    # Get Guassian Values
    for kx in range(kernalSize):
        for ky in range(kernalSize):
            x = kx - radius
            y = ky - radius
            n = math.pow(math.e, -(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * math.pi * (sigma ** 2))
            k_sum += n
            kernal[kx, ky] = n
    # Normalize Guassian Values
    for kx in range(kernalSize):
        for ky in range(kernalSize):
            kernal[kx, ky] = kernal[kx, ky] / k_sum
    return kernal
            
kernal = getKernal(6, 3)
sum = 0
for row in kernal:
    for val in row: sum += val
    print('  '.join(f'{val:5.4f}' for val in row))
print(f"{sum} ~= 1")