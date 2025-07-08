from init import *

def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        logging.info(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        logging.info(f"Inf detected in {name}")

def calculate_cnn_output_size(n_layers, input_length, kernel_size,
                              dilation, padding=0, stride=1):
    """ Assumed that the Pooling layer always follows a CNN layer. """
    seq_len = input_length
    for _ in range(n_layers):
        seq_len = (seq_len - dilation*(kernel_size - 1) - padding - 1) // kernel_size + 1
        seq_len = (seq_len - kernel_size) * stride // kernel_size + 1
    return seq_len