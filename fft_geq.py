"""
FFT-based Graphic EQ implementation
Processes audio blocks using FFT-based filtering
"""
import numpy as np

import settings

class FFTGraphicEQ:
    """FFT-based Graphic Equalizer implementation."""

    def __init__(self, sample_rate=settings.SAMPLE_RATE, fft_size=settings.FFT_SIZE):
        """Initialize the FFT-based graphic EQ.
        
        Args:
            sample_rate: Sample rate in Hz
            fft_size: FFT size (should be a power of 2)
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        self.band_gains = []  # Will store the gain values for each band
        self.band_centers = []  # Will store the center frequencies for each band
        
    def set_bands(self, band_centers):
        """Set the EQ band center frequencies.
        
        Args:
            band_centers: List of center frequencies for the EQ bands
        """
        self.band_centers = band_centers
        self.band_gains = [0.0] * len(band_centers)  # Initialize with 0 dB gain
        
    def set_gain(self, band_idx, gain_db):
        """Set the gain for a specific EQ band.
        
        Args:
            band_idx: Index of the band to adjust
            gain_db: Gain value in dB
        """
        if 0 <= band_idx < len(self.band_gains):
            self.band_gains[band_idx] = gain_db
            
    def set_all_gains(self, gains):
        """Set all EQ band gains at once.
        
        Args:
            gains: List of gain values in dB, one for each band
        """
        if len(gains) == len(self.band_gains):
            self.band_gains = gains.copy()
        else:
            raise ValueError(f"Expected {len(self.band_gains)} gain values, got {len(gains)}")
            
    def process_block(self, input_block):
        """Process an audio block using FFT-based EQ.
        
        Args:
            input_block: 1D numpy array of audio samples
            
        Returns:
            Processed audio block with EQ applied
        """
        # Pad input if needed
        if len(input_block) < self.fft_size:
            input_padded = np.zeros(self.fft_size)
            input_padded[:len(input_block)] = input_block
        else:
            input_padded = input_block[:self.fft_size]
            
        # Compute FFT
        fft = np.fft.rfft(input_padded)
        fft_mag = np.abs(fft)
        fft_phase = np.angle(fft)
        
        # Apply EQ bands in frequency domain
        for i, center_freq in enumerate(self.band_centers):
            if self.band_gains[i] == 0:
                continue
                
            gain_db = self.band_gains[i]
            gain_linear = 10**(gain_db / 20.0)  # Convert dB to linear
            
            # Simple bell curve EQ
            bandwidth = center_freq * 0.7  # Width of the band
            
            # Calculate frequency response for this band using a Gaussian-like curve
            freq_resp = 1 + (gain_linear - 1) * np.exp(
                -((self.frequencies - center_freq) ** 2) / (2 * (bandwidth ** 2))
            )
            
            # Apply to FFT magnitude
            fft_mag *= freq_resp
            
        # Reconstruct complex FFT
        fft_new = fft_mag * np.exp(1j * fft_phase)
        
        # Inverse FFT
        output = np.fft.irfft(fft_new)
        
        # Trim to original length if needed
        if len(output) > len(input_block):
            output = output[:len(input_block)]
            
        return output
        
    def reset_states(self):
        """Reset any internal states (not really needed for FFT implementation)."""
        pass  # No persistent state to reset for basic FFT processing