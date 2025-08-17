"""
DSP Engine - Signal Processing for the Equalizer
- Implements FIR, IIR, FFT-based equalizers
- Applies low-cut and high-cut filters
- Takes EQ band settings and input signal → returns output signal
"""
import numpy as np
from scipy import signal
import settings
from backend.fir_geq import FIRGraphicEQ  
from backend.iir_geq import IIRGraphicEQ
from backend.fft_geq import FFTGraphicEQ  # Add this import

class DSPEngine:
    """Implements signal processing for the equalizer."""
    
    def __init__(self, eq_model):
        """Initialize the DSP engine with the EQ model."""
        self._model = eq_model
        self._sample_rate = settings.SAMPLE_RATE
        self._fft_size = 2048
        self.fir_geq = FIRGraphicEQ(sample_rate=self._sample_rate, num_taps=settings.FIR_TAP_NUMBER)
        self.iir_geq = IIRGraphicEQ(sample_rate=self._sample_rate)
        self.fft_geq = FFTGraphicEQ(sample_rate=self._sample_rate, fft_size=self._fft_size)
        
        # Set band center frequencies for FFT EQ
        band_centers = [band["center"] for band in self._model.eq_bands]
        self.fft_geq.set_bands(band_centers)
        
        self._previous_eq_values = None  # Track previous EQ values for FIR
        self._previous_eq_values_iir = None  # Track previous EQ values for IIR
        self._previous_eq_values_fft = None  # Track previous EQ values for FFT
        self._current_dsp_type = None  # Track the current DSP type for change detection

    def process_audio(self, input_buffer):
        """Process audio based on the current EQ settings.
        Args:
            input_buffer: 1D NumPy array of audio samples 
        """
        # Check if DSP type has changed
        if self._model.dsp_type != self._current_dsp_type:
            # print(f"DSP type changed: {self._current_dsp_type} → {self._model.dsp_type}")
            self._current_dsp_type = self._model.dsp_type
            
            # Reset filter states when changing DSP algorithms
            self.reset_filter_states()

            # Update notification via the model's comprehensive status
            if hasattr(self._model, 'get_status_summary') and hasattr(self._model, 'set_notification'):
                status_summary = self._model.get_status_summary()
                formatted_text = self._model.format_status_summary(status_summary)
                self._model.set_notification(formatted_text)
        
        if input_buffer is None or (isinstance(input_buffer, (list, np.ndarray)) and len(input_buffer) == 0):
            return []
            
        # Convert to numpy array if needed
        if isinstance(input_buffer, list):
            input_buffer = np.array(input_buffer)
            
        # Ensure input is always 1D for processing
        if input_buffer.ndim > 1:
            input_buffer_1d = input_buffer[:, 0]  # Extract channel for processing
        else:
            input_buffer_1d = input_buffer
    
        # Apply selected algorithm
        if self._model.dsp_type == "Bypass":
            # Just pass through the input signal without processing
            output = input_buffer_1d
        elif self._model.dsp_type == "FIR":
            output = self._process_fir_geq(input_buffer_1d)

            # debug
            # samplerate = 48000        # Hz
            # num_samples = 1024
            # frequency = 440           # Hz (A4 tone)
            # t = np.arange(num_samples) / samplerate     # time array, 1024 samples
            # signal = 0.5 * np.sin(2 * np.pi * frequency * t)  # amplitude 0.5
            # output = signal.reshape(-1, 1)  # shape (1024, 1)

        elif self._model.dsp_type == "IIR":
            output = self._process_iir_geq(input_buffer_1d)
        elif self._model.dsp_type == "FFT":
            output = self._process_fft(input_buffer_1d)
        else:
            output = input_buffer_1d  # Fallback

        # Debugging output
        # print(f"Processing with DSP type: {self._model.dsp_type}, input length: {len(input_buffer)}, output length: {len(output)}")
            
        # Apply global filters if not in bypass mode
        if self._model.dsp_type != "Bypass":
            if self._model.low_cut_enabled:
                output = self._apply_low_cut(output)
            if self._model.high_cut_enabled:
                output = self._apply_high_cut(output)
            if self._model.denoise:
                output = self._apply_denoise(output)

            # Apply master gain (convert dB to linear)
            gain_db = getattr(self._model, "master_gain", 0)
            gain_linear = 10 ** (gain_db / 20.0)
            output *= gain_linear

        # Ensure we don't clip
        output = np.clip(output, -1.0, 1.0)
        
        # Return as list for the model
        if input_buffer.ndim > 1:
            return output.reshape(-1, 1)
        else:
            return output  # Return NumPy array directly
    
    def _process_fir_geq(self, input_block):
        """
        Process audio using FIR graphic EQ.
        Assumes input_block is 1D (mono) since stereo handling is done in _duplex_audio_callback.
        """
        if len(input_block) == 0:
            return input_block
        
        # Get current EQ band values
        current_values = [band["value"] for band in self._model.eq_bands]
        
        # Only update gains if values changed
        if self._previous_eq_values is None or current_values != self._previous_eq_values:
            self.fir_geq.set_all_gains(current_values)
            self._previous_eq_values = current_values.copy()  # Store a copy
    
        # Process with current gain settings
        output = self.fir_geq.process_block(input_block)
        
        # Ensure output is within [-1.0, 1.0] range
        output = np.clip(output, -1.0, 1.0)
        
        return output



    def _design_peaking_filter(self, center_freq, gain, Q=1.0):
        """Design peaking (bell) IIR filter coefficients."""
        w0 = 2 * np.pi * center_freq / self._sample_rate
        alpha = np.sin(w0) / (2 * Q)
        A = 10**(gain / 40.0)  # Convert dB to linear

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1, a2]) / a0
        return b, a

    def _process_iir_geq(self, input_block):
        """
        Process audio using IIR graphic EQ.
        Assumes input_block is 1D (mono) since stereo handling is done in _duplex_audio_callback.
        """
        if len(input_block) == 0:
            return input_block
        
        # Debugging - check input signal
        # print(f"IIR input signal range: {np.min(input_block)} to {np.max(input_block)}")
        
        # Get current EQ band values
        current_values = [band["value"] for band in self._model.eq_bands]
        
        # Only update gains if values changed
        if self._previous_eq_values_iir is None or current_values != self._previous_eq_values_iir:
            self.iir_geq.set_all_gains(current_values)
            self._previous_eq_values_iir = current_values.copy()
            
        # Process with current gain settings
        try:
            output = self.iir_geq.process_block(input_block)
            
            # Check for NaN and replace with zeros if needed
            if np.isnan(output).any():
                print("WARNING: NaN values detected in IIR output, replacing with zeros")
                output = np.nan_to_num(output, nan=0.0)
                
            # Debugging - check output signal
            # print(f"IIR output signal range: {np.min(output)} to {np.max(output)}")
            
        except Exception as e:
            print(f"Error in IIR processing: {e}")
            output = input_block.copy()  # Fallback to input
    
        # Ensure output is within [-1.0, 1.0] range
        output = np.clip(output, -1.0, 1.0)
        
        return output

    def _process_fft(self, input_buffer):
        """Process audio using FFT-based filtering."""
        if len(input_buffer) == 0:
            return input_buffer
        
        # Get current EQ band values
        current_values = [band["value"] for band in self._model.eq_bands]
        
        # Only update gains if values changed
        if self._previous_eq_values_fft is None or current_values != self._previous_eq_values_fft:
            self.fft_geq.set_all_gains(current_values)
            self._previous_eq_values_fft = current_values.copy()
        
        # Process with current gain settings
        output = self.fft_geq.process_block(input_buffer)
        
        # Ensure output is within [-1.0, 1.0] range
        output = np.clip(output, -1.0, 1.0)
        
        return output
        
    def _apply_low_cut(self, input_buffer):
        """Apply a high-pass filter (low cut)."""
        # Simple high-pass filter
        cutoff = 80  # Hz
        nyquist = self._sample_rate / 2.0
        normal_cutoff = cutoff / nyquist
        
        # 4th order Butterworth filter
        b, a = signal.butter(4, normal_cutoff, btype='highpass')
        return signal.filtfilt(b, a, input_buffer)
        
    def _apply_high_cut(self, input_buffer):
        """Apply a low-pass filter (high cut)."""
        # Simple low-pass filter
        cutoff = 18000  # Hz
        nyquist = self._sample_rate / 2.0
        normal_cutoff = cutoff / nyquist
        
        # 4th order Butterworth filter
        b, a = signal.butter(4, normal_cutoff, btype='lowpass')
        return signal.filtfilt(b, a, input_buffer)
        
    def _apply_denoise(self, input_buffer):
        """Apply simple noise reduction."""
        # Simple noise gate
        threshold = 0.01
        output = input_buffer.copy()
        output[np.abs(output) < threshold] = 0
        return output

    def reset_filter_states(self):
        """Reset all filter states completely."""
        if hasattr(self, 'fir_geq'):
            self.fir_geq.reset_states()
        
        if hasattr(self, 'iir_geq'):
            self.iir_geq.reset_states()
            
        if hasattr(self, 'fft_geq'):
            self.fft_geq.reset_states()
        
        # Reset any other filter states
        self._zi_lowcut = None
        self._zi_highcut = None
        
        # Reset any FFT overlap buffers
        self._prev_output = np.zeros(64)
