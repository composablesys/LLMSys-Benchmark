import torch
import torch.nn.functional as F

def weighted_pu_loss(y, yhat, mix_stft, prior=0.5, p=1):
    """
    The weighted loss for learning from positive and unlabelled data (PU learning). 
    
    Args:
        y: A mask indicating whether each time-frequency component is positive (1) or unlabelled (0).
        yhat: The output of the convolutional neural network before the non-linear activation function.
        mix_stft: The noisy speech in the short-time Fourier transform domain.
        prior: The class prior for the positive class.
        p: The exponent for the weight. p = 1.0 corresponds to weighting by the magnitude spectrogram of the 
            input noisy speech. p = 0.0 corresponds to no weighting.
        
    Returns:
        The weighted loss.
    """
    # Ensure that the inputs are torch tensors
    y = torch.tensor(y, dtype=torch.float32)
    yhat = torch.tensor(yhat, dtype=torch.float32)
    mix_stft = torch.tensor(mix_stft, dtype=torch.float32)
    
    # Compute the magnitude spectrogram
    w_x = torch.abs(mix_stft)**p
    
    # Compute the weighted sigmoid loss
    pos_loss = w_x * F.binary_cross_entropy_with_logits(yhat, y, reduction='none')
    
    # Compute the three terms for the nnPU loss
    term1 = torch.mean(pos_loss * (y == 1).float())  # Average loss over all positive examples
    term2 = torch.mean(pos_loss * (y == 0).float())  # Average loss over all unlabelled examples
    term3 = prior * term1  # Weighted average loss over all positive examples
    
    # Compute the non-negative PU loss
    loss = term1 + torch.clamp(term2 - term3, min=0)
    
    return loss