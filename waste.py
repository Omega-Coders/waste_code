import torch
from superpoint.models import SuperPointNet

def extract_superpoint_keypoints(image, superpoint, keep_k_best=None, keep_prob=None, remove_borders=4):
    # Convert the image to a tensor and normalize the pixel values to [0, 1]
    tensor_image = torch.from_numpy(image)[None, None, :, :]
    tensor_image = tensor_image.float() / 255.0

    # Pass the image through the SuperPoint model to get the keypoints and descriptors
    with torch.no_grad():
        outs = superpoint({'image': tensor_image})
    keypoints = outs['keypoints'][0].cpu().numpy()
    scores = outs['scores'][0].cpu().numpy()
    descriptors = outs['descriptors'][0].cpu().numpy()

    # Keep only the top k keypoints with highest scores
    if keep_k_best is not None:
        top_k_indices = scores.argsort()[::-1][:keep_k_best]
        keypoints = keypoints[top_k_indices]
        scores = scores[top_k_indices]
        descriptors = descriptors[top_k_indices]

    # Keep only keypoints with a probability above a threshold
    if keep_prob is not None:
        prob_threshold = np.percentile(scores, (1 - keep_prob) * 100)
        selected_indices = np.where(scores >= prob_threshold)[0]
        keypoints = keypoints[selected_indices]
        scores = scores[selected_indices]
        descriptors = descriptors[selected_indices]

    # Remove keypoints near the image borders
    border = remove_borders
    border_mask = np.logical_and(
        np.logical_and(keypoints[:, 0] >= border,
                       keypoints[:, 0] < image.shape[1] - border),
        np.logical_and(keypoints[:, 1] >= border,
                       keypoints[:, 1] < image.shape[0] - border))
    keypoints = keypoints[border_mask]
    scores = scores[border_mask]
    descriptors = descriptors[border_mask]

    return keypoints, descriptors, scores
