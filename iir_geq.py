import numpy as np
from scipy.signal import lfilter, butter, sosfreqz
from math import pi, sin, cos, sqrt

import settings

class IIRGraphicEQ:
    def __init__(self, sample_rate=settings.SAMPLE_RATE, band_centers=None):
        """
        Initialize a 7-band IIR graphic equalizer.
        :param sample_rate: Sampling rate of the audio (Hz).
        :param band_centers: Optional list or array of 7 band center frequencies (Hz).
        """
        # Define 7 default center frequencies if none provided (log-spaced 63 Hz to 16 kHz example)
        if band_centers is None:
            band_centers = settings.EQ_BAND_7_F_CENTERS
        self.fs = sample_rate
        self.band_centers = np.array(band_centers, dtype=float)
        self.num_bands = len(self.band_centers)
        
        # Initialize all band gains to 0 dB (linear gain = 1.0)
        self.band_gains = np.ones(self.num_bands)
        # Compute initial filter coefficients for flat (0 dB) state
        self._design_filters()
        # Initialize filter states for each biquad (for lfilter); each state has length = order (2)
        self._states = [np.zeros(len(a) - 1) for a in self.a_filters]
    
    def _design_filters(self):
        """Design biquad filter coefficients for all 7 bands based on current band_gains."""
        nyquist = 0.5 * self.fs
        # Calculate band edge (crossover) frequencies using geometric means
        band_edges = [20.0]  # start near 20 Hz as the lower bound of first band
        for i in range(self.num_bands - 1):
            f_lower = self.band_centers[i]
            f_upper = self.band_centers[i + 1]
            band_edges.append(sqrt(f_lower * f_upper))
        band_edges.append(nyquist)  # upper bound of the last band is Nyquist
        
        # Lists for filter coefficients
        self.b_filters = []  # numerator (b) coeffs for each band
        self.a_filters = []  # denominator (a) coeffs for each band
        
        # Band 1: Low-shelf filter (boost/cut frequencies below band_edges[1])
        cutoff1 = band_edges[1] / nyquist  # normalized cutoff (0-1)
        b, a = self._design_lowshelf(cutoff1, self.band_gains[0], Q=0.7071)
        self.b_filters.append(b)
        self.a_filters.append(a)
        
        # Bands 2-6: Peaking filters
        for band_index in range(1, self.num_bands - 1):
            center_freq = self.band_centers[band_index]
            # Compute bandwidth such that the filter spans between the adjacent crossover frequencies
            f_lower_edge = band_edges[band_index]
            f_upper_edge = band_edges[band_index + 1]
            bandwidth_hz = f_upper_edge - f_lower_edge
            # Calculate Q = center_freq / bandwidth_hz (approximation for constant-Q)
            Q = 1.0
            if bandwidth_hz > 0:
                Q = center_freq / bandwidth_hz
            # Design peaking filter for this band
            w0 = center_freq / nyquist  # normalized center frequency (0-1)
            b, a = self._design_peaking(w0, self.band_gains[band_index], Q=Q)
            self.b_filters.append(b)
            self.a_filters.append(a)
        
        # Band 7: High-shelf filter (boost/cut frequencies above band_edges[-2])
        cutoff7 = band_edges[-2] / nyquist  # normalized cutoff for high shelf
        b, a = self._design_highshelf(cutoff7, self.band_gains[-1], Q=0.7071)
        self.b_filters.append(b)
        self.a_filters.append(a)
    
    def _design_peaking(self, norm_center_freq, gain_lin, Q=1.0):
        """
        Design a peaking (bell) filter at normalized frequency norm_center_freq (0-1 relative to Nyquist).
        gain_lin is the linear gain (1.0 = 0 dB), Q is the quality factor.
        Returns (b, a) filter coefficient arrays.
        """
        # Limit the frequency to (0, 1) range (not inclusive) to avoid extreme values
        w0 = max(1e-6, min(norm_center_freq, 1 - 1e-6)) * pi  # convert to radians
        A = sqrt(max(1e-6, gain_lin))        # A = sqrt(gain) for use in formula
        Q = max(1e-6, Q)                     # ensure Q is positive and not too small
        alpha = sin(w0) / (2 * Q)
        
        # Biquad peaking filter formula (RBJ Cookbook)
        b0 = 1 + alpha * A
        b1 = -2 * cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos(w0)
        a2 = 1 - alpha / A
        # Normalize to make a0 = 1
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        return b, a
    
    def _design_lowshelf(self, norm_cutoff, gain_lin, Q=0.7071):
        """
        Design a low-shelf filter with normalized cutoff frequency (0-1) and linear gain.
        Q=0.7071 gives a gentle 2nd-order shelf slope.
        """
        w0 = max(1e-6, min(norm_cutoff, 1 - 1e-6)) * pi
        A = sqrt(max(1e-6, gain_lin))
        Q = max(1e-6, Q)
        alpha = sin(w0) / (2 * Q)
        
        # Low shelf formula (RBJ Cookbook)
        b0 = A * ((A + 1) - (A - 1) * cos(w0) + 2 * sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos(w0))
        b2 = A * ((A + 1) - (A - 1) * cos(w0) - 2 * sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos(w0) + 2 * sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos(w0))
        a2 = (A + 1) + (A - 1) * cos(w0) - 2 * sqrt(A) * alpha
        # Normalize coefficients
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        return b, a
    
    def _design_highshelf(self, norm_cutoff, gain_lin, Q=0.7071):
        """
        Design a high-shelf filter with normalized cutoff frequency (0-1) and linear gain.
        """
        w0 = max(1e-6, min(norm_cutoff, 1 - 1e-6)) * pi
        A = sqrt(max(1e-6, gain_lin))
        Q = max(1e-6, Q)
        alpha = sin(w0) / (2 * Q)
        
        # High shelf formula (RBJ Cookbook)
        b0 = A * ((A + 1) + (A - 1) * cos(w0) + 2 * sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos(w0))
        b2 = A * ((A + 1) + (A - 1) * cos(w0) - 2 * sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos(w0) + 2 * sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos(w0))
        a2 = (A + 1) - (A - 1) * cos(w0) - 2 * sqrt(A) * alpha
        # Normalize coefficients
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        return b, a
    
    def set_band_gain(self, band_index, gain_db):
        """
        Set the gain of a single band (indexed 0-6) in decibels, and update that filter's coefficients.
        """
        if not (0 <= band_index < self.num_bands):
            raise IndexError("Band index %d out of range for %d bands." % (band_index, self.num_bands))
        # Convert dB gain to linear scale
        new_gain = 10.0 ** (gain_db / 20.0)
        self.band_gains[band_index] = new_gain
        # Recompute the coefficients for this band
        nyquist = 0.5 * self.fs
        # Recompute band edge frequencies (needed for bandwidth/Q calculation)
        band_edges = [20.0]
        for i in range(self.num_bands - 1):
            band_edges.append(sqrt(self.band_centers[i] * self.band_centers[i + 1]))
        band_edges.append(nyquist)
        # Design appropriate filter type for the band
        if band_index == 0:
            # Low shelf for band 0
            cutoff1 = band_edges[1] / nyquist
            b, a = self._design_lowshelf(cutoff1, new_gain, Q=0.7071)
        elif band_index == self.num_bands - 1:
            # High shelf for last band
            cutoff7 = band_edges[-2] / nyquist
            b, a = self._design_highshelf(cutoff7, new_gain, Q=0.7071)
        else:
            # Peaking filter for mid band
            center_freq = self.band_centers[band_index]
            f_lower_edge = band_edges[band_index]
            f_upper_edge = band_edges[band_index + 1]
            bw = f_upper_edge - f_lower_edge
            Q = center_freq / bw if bw > 0 else 1.0
            w0 = center_freq / nyquist
            b, a = self._design_peaking(w0, new_gain, Q=Q)
        # Update the coefficients and reset that filter's state
        self.b_filters[band_index] = b
        self.a_filters[band_index] = a
        self._states[band_index] = np.zeros(len(a) - 1)
    
    def set_all_gains(self, gains_db):
        """
        Set gains for all bands at once using a list of 7 gains (in dB), then redesign all filters.
        """
        if len(gains_db) != self.num_bands:
            raise ValueError("Expected %d gain values (one per band), got %d." % (self.num_bands, len(gains_db)))
        # Update internal gains (linear)
        self.band_gains = np.array([10.0 ** (g/20.0) for g in gains_db], dtype=float)
        # Redesign all filter coefficients with the new gains
        self._design_filters()
        # Reset all filter states (since coefficients changed)
        self._states = [np.zeros(len(a) - 1) for a in self.a_filters]
    
    def process_block(self, input_signal):
        """
        Process an input audio signal (1-D NumPy array) through the 7-band EQ.
        Returns a new NumPy array with the EQ applied.
        """
        # Ensure input is a 1-D array (mono signal). If stereo or multi-channel, take first channel.
        x = np.array(input_signal, dtype=float)
        if x.ndim > 1:
            x = x[:, 0]
        y = x.copy()
        # Apply each band's filter in sequence (cascaded)
        for i in range(self.num_bands):
            # Apply IIR filter with internal state to maintain continuity
            y, self._states[i] = lfilter(self.b_filters[i], self.a_filters[i], y, zi=self._states[i])
        return y
    
    def reset_states(self):
        """Reset internal filter states (e.g., when processing a new, unrelated audio stream)."""
        self._states = [np.zeros(len(a) - 1) for a in self.a_filters]
