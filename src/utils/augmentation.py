from scipy.signal import firwin, lfilter
import torch


# FIR filter design
def design_fir_filter(filter_type, numtaps=101, cutoff=0.3):
    if filter_type == 'lowpass':
        fir_coeff = firwin(numtaps, cutoff=cutoff, window='hamming')
    elif filter_type == 'highpass':
        fir_coeff = firwin(numtaps, cutoff=cutoff, window='hamming', pass_zero=False)
    elif filter_type == 'bandpass':
        fir_coeff = firwin(numtaps, cutoff=[cutoff[0], cutoff[1]], window='hamming', pass_zero=False)
    else:
        raise ValueError("Invalid filter type. Choose from 'lowpass', 'highpass', 'bandpass'.")
    return fir_coeff

# Apply FIR filter to waveform
def apply_fir_filter(waveform, fir_coeff):
    filtered_waveform = lfilter(fir_coeff, 1.0, waveform)
    return torch.from_numpy(filtered_waveform).float()

# Data augmentation function
def augment_waveform(waveform, filter_type=None, nb_cutoff=0.3, wb_cutoff=[0.2, 0.7]):
    if filter_type == 'nb':
        fir_coeff = design_fir_filter('lowpass', cutoff=nb_cutoff)
        augmented_waveform = apply_fir_filter(waveform, fir_coeff)
    elif filter_type == 'wb':
        fir_coeff = design_fir_filter('bandpass', cutoff=wb_cutoff)
        augmented_waveform = apply_fir_filter(waveform, fir_coeff)
    else:
        augmented_waveform = waveform  # No augmentation
    return augmented_waveform
