def min_dist_scaled(points, res, chunk_size=1024):
    """
    Scalable version of min_dist to handle very large resolutions.

    Inputs:
      * points: Tensor of shape [B, P, 2], where B is the batch size, P is the number of points.
      * res: Resolution of the output grid.
      * chunk_size: Number of grid points to process in each chunk to limit memory usage.
    Returns:
      Tensor of shape [B, res, res] with distances to the nearest point.
    """
    batch_size, num_points, _ = points.shape

    # Create a grid for the output
    x = torch.linspace(0, 1, res)
    y = torch.linspace(0, 1, res)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # Shape: (res^2, 2)

    # Prepare output tensor
    output = torch.empty(batch_size, res * res, dtype=torch.float32)

    # Process the grid in chunks
    for i in range(0, grid.shape[0], chunk_size):
        chunk = grid[i:i + chunk_size].unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (B, chunk_size, 2)
        distances = torch.cdist(chunk, points)  # Shape: (B, chunk_size, P)
        min_distances = distances.min(dim=2).values  # Shape: (B, chunk_size)
        output[:, i:i + chunk_size] = min_distances

    # Reshape to [B, res, res]
    distance_tensor = output.view(batch_size, res, res)
    return distance_tensor

# Test Part 3-2: Scalable Version
batch_size = 3
num_points = 4
res = 16384  # Very high resolution
chunk_size = 1024  # Process in manageable chunks

points = create_points(batch_size, num_points)
distance_maps = min_dist_scaled(points, res, chunk_size)

# Visualize a lower-resolution version for clarity
low_res = 256
distance_maps_low = min_dist_scaled(points, low_res, chunk_size)

plt.figure(figsize=(8, 8))
plt.imshow(distance_maps_low[0].numpy(), cmap="viridis")
plt.colorbar(label="Distance to Nearest Point")
plt.title("Distance Map for High Resolution (Scaled Down for Visualization)")
plt.show()
