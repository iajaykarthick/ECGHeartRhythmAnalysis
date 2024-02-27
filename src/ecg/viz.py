import numpy as np
import matplotlib.pyplot as plt


def plot_ecg(ecg_signals, overlaid=False, subplot=False, title=None, labels=None):
    
    # if ecg_signals is a list of lists or a 2D array or list of np arrays, then it's a list of signals
    if isinstance(ecg_signals[0], (list, np.ndarray)):
        ecg_signals = np.array(ecg_signals)
    else:
        ecg_signals = np.array([ecg_signals])
    
    if title is None:
        title = 'ECG Signals' if ecg_signals.shape[0] > 1 else 'ECG Signal'
    
    if overlaid:
        plt.figure(figsize=(10, 4))
        for i, ecg_signal in enumerate(ecg_signals):
            plt.plot(ecg_signal, label=f'Signal {i+1}')
        plt.title('Overlaid ECG Signals')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (samples)')
        if labels and len(labels) == ecg_signals.shape[0]:
            plt.legend(labels)
        else:
            plt.legend()
        plt.grid(True)
        plt.show()
    elif ecg_signals.shape[0] == 1: # Single signal
        ecg_signal = ecg_signals[0]
        plt.figure(figsize=(10, 4))
        plt.plot(ecg_signal)
        plt.title('ECG Signal')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (samples)')
        plt.grid(True)
        plt.show()    
    else: # Default to subplot if not overlaid and list of signals
        fig, axs = plt.subplots(len(ecg_signals), 1, figsize=(10, 4*len(ecg_signals)))
        for i, ecg_signal in enumerate(ecg_signals):
            axs[i].plot(ecg_signal)
            axs[i].set_title(f'ECG Signal {i+1}')
            axs[i].set_ylabel('Amplitude')
            axs[i].grid(True)
        axs[-1].set_xlabel('Time (samples)')
        plt.tight_layout()
        plt.show()