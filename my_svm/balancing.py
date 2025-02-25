import numpy as np

def undersample(feature_matrix, labels):
    unique_classes, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)
    resampled_indices = []

    for cls in unique_classes:
        indices = np.where(labels == cls)[0]
        if len(indices) > min_count:
            resampled_indices.extend(np.random.choice(indices, min_count, replace=False))
        else:
            resampled_indices.extend(indices)

    shuffle_indices = np.arange(len(resampled_indices))
    np.random.shuffle(shuffle_indices)

    return feature_matrix[resampled_indices], labels[resampled_indices]
