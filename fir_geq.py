import numpy as np
from scipy.signal import firwin, lfilter

import settings


class FIRGraphicEQ:
    def __init__(self, sample_rate=48000, num_taps=settings.FIR_TAP_NUMBER):
        """7-band graphic equalizer with linear-phase FIR filters."""
        self.fs = sample_rate
        self.num_taps = num_taps
        self.band_centers = settings.EQ_BAND_7_F_CENTERS

        # Calculate band edge frequencies (geometric means between centers)
        # We add a low cutoff (~20 Hz) and high cutoff (Nyquist) at ends.
        edges = [20.0]  # start ~20 Hz for Band 1
        for i in range(len(self.band_centers) - 1):
            f_lower = self.band_centers[i]
            f_upper = self.band_centers[i+1]
            edges.append(np.sqrt(f_lower * f_upper))
        edges.append(sample_rate / 2.0)  # Nyquist for Band 7

        # Design linear-phase FIR filters for each band
        self.filters = []
        # Band 1 (low-pass from DC to edges[1])
        b1 = firwin(num_taps, edges[1], pass_zero=True, fs=sample_rate)
        self.filters.append(b1)
        # Bands 2–6 (band-pass between adjacent edges)
        for k in range(1, 6):
            b = firwin(num_taps, [edges[k], edges[k+1]], pass_zero='bandpass', fs=sample_rate)
            self.filters.append(b)
        # Band 7 (high-pass from edges[6] to Nyquist)
        b7 = firwin(num_taps, edges[6], pass_zero=False, fs=sample_rate)
        self.filters.append(b7)

        # Initialize filter states (delay lines) for each band: length = num_taps-1
        self._states = [np.zeros(len(b) - 1) for b in self.filters]
        # Initialize gains (linear scale factors) for each band
        self.gains = [1.0] * 7  # start with 0 dB gain (factor 1.0)

    def set_gain(self, band_index, gain_db):
        """Set gain for a specific band index (0-6) in dB."""
        if 0 <= band_index < 7:
            self.gains[band_index] = 10.0 ** (gain_db / 20.0)

    def set_all_gains(self, gain_db_list):
        """Set gains for all bands at once (provide 7 gains in dB)."""
        if len(gain_db_list) == 7:
            self.gains = [10.0 ** (g/20.0) for g in gain_db_list]

    def process_block(self, input_block_1d):
        """
        Process one block of audio samples.
        Assuming input is 1-D NumPy array (mono audio).
        If input is 2-D (e.g., stereo), it will be flattened to 1-D.
        Returns a 1-D NumPy array of processed samples.
        """
        # Check input dimensions and convert to 1D if needed
        original_shape = input_block_1d.shape
        is_2d = len(original_shape) > 1
        
        # If input is 2D (stereo), flatten to 1D for processing
        if is_2d:
            # Extract first channel only - assuming mono processing
            input_block_1d = input_block_1d[:, 0]
        
        # Create output with same type as flattened input
        output_block = np.zeros_like(input_block_1d, dtype=float)
        
        # Process each band filter and accumulate
        for i, b in enumerate(self.filters):
            # Apply FIR filter for band i on the input block, with initial state
            y, zf = lfilter(b, 1.0, input_block_1d, zi=self._states[i])
            # Save final state for next block
            self._states[i] = zf
            # Mix the band's contribution into output with the band's gain
            output_block += self.gains[i] * y
        
        return output_block

    def reset_states(self):
        """Reset filter states (for processing new channels)."""
        self._states = [np.zeros(len(b) - 1) for b in self.filters]
