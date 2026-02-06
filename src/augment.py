
import numpy as np

def augment_stroke(points, rotation_range=10, scale_range=0.1, jitter_std=0.0):
    """
    Applies random augmentation to a stroke (sequence of points).
    
    Args:
        points (np.ndarray): Shape (N, 2) or (N, D).
        rotation_range (float): Max rotation in degrees.
        scale_range (float): Max scaling factor deviation (e.g. 0.1 means 0.9 to 1.1).
        jitter_std (float): Standard deviation for Gaussian noise added to points.
        
    Returns:
        np.ndarray: Augmented points.
    """
    if len(points) == 0:
        return points
        
    augmented = points.copy()
    
    # 1. Rotation
    if rotation_range > 0:
        angle_deg = np.random.uniform(-rotation_range, rotation_range)
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Rotate around center of mass
        center = np.mean(augmented[:, :2], axis=0)
        augmented[:, :2] = np.dot(augmented[:, :2] - center, rotation_matrix.T) + center

    # 2. Scaling
    if scale_range > 0:
        scale_factor = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
        # Scale around center
        center = np.mean(augmented[:, :2], axis=0)
        augmented[:, :2] = (augmented[:, :2] - center) * scale_factor + center
        
    # 3. Jitter (Noise)
    if jitter_std > 0:
        noise = np.random.normal(0, jitter_std, augmented[:, :2].shape)
        augmented[:, :2] += noise
        
    return augmented
