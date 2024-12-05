import torch
import numpy as np
import matplotlib.pyplot as plt

def create_points(batch_size, num_points):
    """
    Create random points for each batch.
    Inputs:
      * batch_size: number of batches.
      * num_points: number of points in each batch.
    Returns:
      A tensor of shape [batch_size, num_points, 2].
    """
    return torch.rand(batch_size, num_points, 2)

def min_dist(points, res):
    """
    Calculates a 3D tensor with the minimum distance from each pixel to data.

    Inputs:
      * points: Tensor of shape [B, P, 2], where B is the batch size, P is the number of points.
      * res: Resolution of the output grid.
    Returns:
      Tensor of shape [B, res, res] with distances to the nearest point.
    """
    batch_size, num_points, _ = points.shape

    # Create a grid for the output
    x = torch.linspace(0, 1, res)
    y = torch.linspace(0, 1, res)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # Shape: (res^2, 2)

    # Expand the grid to match the batch size
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (B, res^2, 2)

    # Compute pairwise distances between grid points and data points
    distances = torch.cdist(grid, points)  # Shape: (B, res^2, P)

    # Find the minimum distance for each grid point
    min_distances = distances.min(dim=2).values  # Shape: (B, res^2)

    # Reshape to [B, res, res]
    distance_tensor = min_distances.view(batch_size, res, res)
    return distance_tensor

# Test 3-1a: Single Batch
points = torch.tensor([[0.4, 0.3], [0.6, 0.7]]).unsqueeze(0)  # Shape [1, 2, 2]
distance_map = min_dist(points, 20)
plt.figure(figsize=(8, 8))
plt.imshow(distance_map[0].numpy(), cmap="viridis")
plt.colorbar(label="Distance to Nearest Point")
plt.title("Distance Map (Single Batch, Resolution 20)")
plt.show()

# Test 3-1b: Batched Version
batch_size = 3
num_points = 4
res = 256
points = create_points(batch_size, num_points)
distance_maps = min_dist(points, res)

# Visualize one batch
plt.figure(figsize=(8, 8))
plt.imshow(distance_maps[0].numpy(), cmap="viridis")
plt.colorbar(label="Distance to Nearest Point")
plt.title("Distance Map for Batch 1 (Resolution 256)")
plt.show()
