import numpy as np

def douglas_peucker(points, epsilon):
    """
    Simplifies the stroke using the Douglas-Peucker algorithm.
    :param points: numpy array of shape (N, 2) or (N, 3+)
    :param epsilon: tolerance
    :return: simplified points
    """
    if len(points) < 3:
        return points

    dmax = 0
    index = 0
    end = len(points) - 1
    
    # Line defined by p1 and p2
    p1 = points[0, :2]
    p2 = points[end, :2]
    
    # If p1 and p2 are the same, distance is just distance to p1
    if np.allclose(p1, p2):
        d = np.linalg.norm(points[1:end, :2] - p1, axis=1)
        if len(d) > 0:
            dmax = np.max(d)
            index = np.argmax(d) + 1
    else:
        # Perpendicular distance
        # vector from p1 to p2
        v = p2 - p1
        # normalize
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        # normal vector
        n = np.array([-v_norm[1], v_norm[0]])
        
        # vectors from p1 to all points
        w = points[1:end, :2] - p1
        
        # project w onto n
        d = np.abs(np.dot(w, n))
        
        if len(d) > 0:
            dmax = np.max(d)
            index = np.argmax(d) + 1

    if dmax > epsilon:
        # Recursive call
        rec_results1 = douglas_peucker(points[:index+1], epsilon)
        rec_results2 = douglas_peucker(points[index:], epsilon)
        
        # concatenate results, skipping the duplicate point
        return np.vstack((rec_results1[:-1], rec_results2))
    else:
        return np.vstack((points[0], points[end]))

def resample(points, num_points):
    """
    Resamples the stroke to a fixed number of points equidistant along the path.
    :param points: numpy array of shape (N, D)
    :param num_points: target number of points (L)
    :return: resampled points (L, D)
    """
    if len(points) <= 1:
        return np.tile(points, (num_points, 1))

    # Calculate cumulative distance
    dists = np.linalg.norm(points[1:, :2] - points[:-1, :2], axis=1)
    cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
    total_length = cum_dists[-1]
    
    if total_length == 0:
        return np.tile(points[0], (num_points, 1))
    
    # Target distances
    target_dists = np.linspace(0, total_length, num_points)
    
    # Interpolate for each dimension
    resampled = np.zeros((num_points, points.shape[1]))
    for i in range(points.shape[1]):
        resampled[:, i] = np.interp(target_dists, cum_dists, points[:, i])
        
    return resampled

def normalize(points):
    """
    Spatial normalization to [0, 1]^2 while preserving aspect ratio and centering.
    :param points: (N, 2) or (N, D) - only first 2 dims are x, y
    :return: normalized points
    """
    coords = points[:, :2]
    min_xy = np.min(coords, axis=0)
    max_xy = np.max(coords, axis=0)
    
    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]
    
    scale = max(width, height)
    if scale == 0:
        return points
        
    # Center and scale
    center = (min_xy + max_xy) / 2
    
    points[:, :2] = (points[:, :2] - center) / scale + 0.5
    
    return points

def extract_features(points):
    """
    Extracts features for each point: [x, y, dx, dy, sin, cos, curvature]
    :param points: (L, 2) normalized points
    :return: features (L, 7)
    """
    L = len(points)
    features = np.zeros((L, 7))
    
    # 1. x, y
    features[:, 0:2] = points[:, :2]
    
    # 2. dx, dy (Forward difference)
    # Pad last point with same delta as previous or 0
    deltas = np.diff(points[:, :2], axis=0)
    # Repeat last delta to keep shape
    if len(deltas) > 0:
        deltas = np.vstack((deltas, deltas[-1]))
    else:
        deltas = np.zeros((L, 2))
        
    features[:, 2:4] = deltas
    
    # 3. sin, cos of segment angle
    # theta = atan2(dy, dx)
    # Note: norms of deltas can be 0 if points are duplicate
    norms = np.linalg.norm(deltas, axis=1) + 1e-8
    
    sin_theta = deltas[:, 1] / norms
    cos_theta = deltas[:, 0] / norms
    
    features[:, 4] = sin_theta
    features[:, 5] = cos_theta
    
    # 4. Curvature (kappa)
    # Using turning angle between segments
    # theta_i is angle of segment i (from point i to i+1)
    # kappa_i is change in angle at point i
    # We can compute theta for each segment, then diff
    
    thetas = np.arctan2(deltas[:, 1], deltas[:, 0])
    
    # diff of thetas. Be careful with wrap around pi/-pi
    # delta_theta = theta_i - theta_{i-1}
    # But usually we want curvature at a point.
    # Let's define curvature at point i as angle change from segment (i-1->i) to (i->i+1)
    
    # Pad theta: theta[-1] is usually same as prev
    
    d_theta = np.diff(thetas)
    # Handle wrap around: if jump is > pi, subtract 2pi. if < -pi, add 2pi
    d_theta = np.mod(d_theta + np.pi, 2 * np.pi) - np.pi
    
    # Pad curvature. First point curvature? usually 0 or same as second.
    # d_theta length is L-1. It represents turning at point 1 to L-1.
    # Point 0 and Point L-1 (last) have undefined curvature by this def.
    # Let's pad with 0 at start and end? Or replicate.
    
    kappa = np.zeros(L)
    if len(d_theta) > 0:
        # d_theta has length L-1.
        # d_theta[0] is turn at point 1.
        # d_theta[L-2] is turn at point L-1.
        kappa[1:] = d_theta
    
    features[:, 6] = kappa
    
    return features

def preprocess_stroke(stroke, L=128, epsilon=1.0):
    """
    Full pipeline
    """
    # 1. Simplify
    # stroke should be (N, 2)
    simplified = douglas_peucker(stroke, epsilon)
    
    # 2. Resample
    resampled = resample(simplified, L)
    
    # 3. Normalize
    normalized = normalize(resampled)
    
    # 4. Features
    feats = extract_features(normalized)
    
    return feats
