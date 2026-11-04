
import numpy as np

def calculate(img, num_samples=500):

    img = img > .5  # Ensure (very naively) the image is binary

    edge_y, edge_x = np.where(img)
    points = np.column_stack((edge_x, edge_y))
    ordered_pts = [points[0]]
    unvisited = list(points[1:])
    while unvisited:
        last_pt = ordered_pts[-1]
        # Calculate Euclidean distances to remaining unvisited edge pixels
        dists = np.sum((unvisited - last_pt)**2, axis=1)
        nearest_idx = np.argmin(dists)
        ordered_pts.append(unvisited.pop(nearest_idx))
    ordered_pts = np.array(ordered_pts)

    distances = np.sqrt(np.sum(np.diff(ordered_pts, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    total_length = cumulative_dist[-1]
    target_dists = np.linspace(0, total_length, num_samples, endpoint=False)

    x_resampled = np.interp(target_dists, cumulative_dist, ordered_pts[:, 0]).astype(int)
    y_resampled = np.interp(target_dists, cumulative_dist, ordered_pts[:, 1]).astype(int)

    return x_resampled, y_resampled
