import torch

def normalize_to_match_peak_batched(signal_quieter, signal_louder):
    """
    Normalize quieter signals to match the peak amplitude of louder signals in a batched input.
    Handles cases where one or both signals in a batch are silent (zero peak amplitude).

    Args:
        signal_quieter (torch.Tensor): The quieter audio signals, shape [B, 1, T].
        signal_louder (torch.Tensor): The louder audio signals, shape [B, 1, T].

    Returns:
        torch.Tensor: The normalized quieter audio signals, shape [B, 1, T].
    """
    # Ensure the tensors are the correct shape
    assert signal_quieter.shape == signal_louder.shape, "Input tensors must have the same shape."
    assert signal_quieter.dim() == 3, "Input tensors must have shape [B, 1, T]."
    
    # Compute peak amplitudes for each signal in the batch
    peak_quieter = torch.max(torch.abs(signal_quieter), dim=2, keepdim=True).values  # Shape: [B, 1, 1]
    peak_louder = torch.max(torch.abs(signal_louder), dim=2, keepdim=True).values    # Shape: [B, 1, 1]
    
    # Handle silent signals
    # Avoid division by zero: Add a small epsilon to denominator where peak_quieter is zero
    epsilon = 1e-8
    scaling_factor = torch.where(peak_quieter > 0, peak_louder / (peak_quieter + epsilon), torch.zeros_like(peak_louder))
    
    # Scale the quieter signals
    signal_quieter_normalized = signal_quieter * scaling_factor
    
    # For fully silent batches (where both peak_louder and peak_quieter are zero), retain the original signal
    all_silent = (peak_louder == 0) & (peak_quieter == 0)
    signal_quieter_normalized = torch.where(all_silent, signal_quieter, signal_quieter_normalized)

    return signal_quieter_normalized