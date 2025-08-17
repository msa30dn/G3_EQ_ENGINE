from settings import EQ_BAND_7_F_CENTERS
import threading
from PySide6.QtCore import QObject, Signal

import settings  # Import Signal for property change notifications

# The Data Layer - Represents the core domain logic of the application.:
# - EQ band values
# -  DSP algorithm: FIR, IIR, FFT
# -  Signal data (input/output)
# -  Global filters (low-cut, high-cut)
# - AI classification result (e.g., "rock song")
# This model is pure: it doesn't know about the GUI. It just holds the data.

class EQModel(QObject):  # Add QObject as parent class
    property_changed = Signal(str, object)  # Signal(property_name, new_value)
    
    def __init__(self):
        super().__init__()  # Call QObject's __init__
        # Add a lock for EQ band access
        self._eq_lock = threading.Lock()
        
        self.eq_bands = [{"center": freq, "value": 0.0} for freq in EQ_BAND_7_F_CENTERS]
        
        # Add a lock for signal buffers
        self._signal_lock = threading.Lock()
        self.input_signal = []
        self.output_signal = []
        
        self.dsp_type = "FIR" # FIR, IIR, FFT
        self.low_cut_enabled = False # high-pass filter
        self.high_cut_enabled = False # low-pass filter
        self.denoise_enabled = False  # or True for testing
        self.master_gain = 0.0  # Default to 0 dB (no gain)
        self.category_enabled = False
        self.category = ""
        self.notification = "Hello! This is a notification from EQModel"  # Added notification property
        self.is_streaming = False

        if settings.DENOISE_PROFILE == 1:
            self.denoise_profile1()
        elif settings.DENOISE_PROFILE == 2:
            self.denoise_profile2()
        else:
            self.my_denoise_profile()

    def denoise_profile1(self):
        """
        Very strong speech denoise.
        Much tighter thresholds, high ratio, deep floor, long hold/release.
        Expect a drier sound and stronger attenuation in pauses.
        """
        # Push thresholds up so the gate closes more often
        self.gate_thresh_open_db  = -26.0
        self.gate_thresh_close_db = -34.0   # 8 dB hysteresis

        # Heavy downward expansion
        self.gate_ratio     = 20.0          # was 12
        self.gate_floor_db  = -96.0         # near mute when closed

        # Longer hold and releases to prevent chatter
        self.gate_hold_ms         = 220.0
        self.gate_env_attack_ms   = 0.8     # fast detector attack to catch onsets
        self.gate_env_release_ms  = 250.0   # slow detector release = stable decision
        self.gate_gain_attack_ms  = 0.6     # opens quickly
        self.gate_gain_release_ms = 300.0   # closes gently to hide pumping
    
    def denoise_profile2(self):
        """
        EXTREME speech denoise.
        Opens only on pretty loud voice; near-mute in pauses.
        """

        # Very high thresholds (open only when voice is strong)
        self.gate_thresh_open_db  = -18.0
        self.gate_thresh_close_db = -30.0   # 12 dB hysteresis

        # Very heavy downward expansion
        self.gate_ratio     = 80.0          # practically a gate
        self.gate_floor_db  = -120.0        # near mute when closed

        # Longer hold and slow releases for rock-solid closure
        self.gate_hold_ms         = 150.0 # 350.0
        self.gate_env_attack_ms   = 0.3     # very fast to catch onsets
        self.gate_env_release_ms  = 200 # 400.0   # slow detector release (stable decision)
        self.gate_gain_attack_ms  = 0.3     # open quickly
        self.gate_gain_release_ms = 210.0 # 420.0   # close gently to hide pumping

    def my_denoise_profile(self):
        """
        MAX suppression (brick-wall vibe).
        Use only when isolation matters more than naturalness.
        """

        # Opens only on very hot speech; stays closed otherwise
        self.gate_thresh_open_db  = -12.0
        self.gate_thresh_close_db = -28.0   # 16 dB hysteresis

        # Essentially hard gate depth
        self.gate_ratio     = 100.0
        self.gate_floor_db  = -144.0        # effectively silence when closed

        # Very assertive timing to prevent chatter
        self.gate_hold_ms         = 200.0   # 500
        self.gate_env_attack_ms   = 0.2
        self.gate_env_release_ms  = 200.0   # 600
        self.gate_gain_attack_ms  = 0.2
        self.gate_gain_release_ms = 200.0   # 600

    def getNotification(self):
        """Return the current notification message"""
        return self.notification
    
    def setNotification(self, message):
        """Set a new notification message"""
        self.notification = message
        
    def get_eq_band_value(self, idx):
        with self._eq_lock:
            if 0 <= idx < len(self.eq_bands):
                return self.eq_bands[idx]["value"]
        return 0.0
        
    def set_eq_band_value(self, idx, value):
        with self._eq_lock:
            if 0 <= idx < len(self.eq_bands):
                self.eq_bands[idx]["value"] = value
                
    def get_status_summary(self):
        """Generate a comprehensive status summary of current settings as JSON."""
        # Get input source with proper formatting
        input_source = "Unknown"
        if hasattr(self, "audio_streamer"):
            source = self.audio_streamer._input_source
            if source == "mic":
                input_source = "Microphone"
            elif source == "system":
                input_source = "System Audio"
            elif source == "file":
                input_source = "File"
        
        # Return as JSON (Python dictionary)
        return {
            "input_source": input_source,
            "dsp_type": self.dsp_type,
            "low_cut": self.low_cut_enabled,
            "high_cut": self.high_cut_enabled,
            "denoise": hasattr(self, "denoise_enabled") and self.denoise_enabled,
            "master_gain": self.master_gain
        }
    
    def format_status_summary(self, status_summary=None):
        """Format the status summary JSON into a readable string."""
        if status_summary is None:
            status_summary = self.get_status_summary()
            
        return (f"Input: {status_summary['input_source']} | "
                f"DSP: {status_summary['dsp_type']} | "
                f"Low Cut: {'On' if status_summary['low_cut'] else 'Off'} | "
                f"High Cut: {'On' if status_summary['high_cut'] else 'Off'} | "
                f"Denoise: {'On' if status_summary['denoise'] else 'Off'} | "
                f"Gain: {status_summary['master_gain']:.1f} dB")
    
    def set_property(self, property_name, value):
        """Set a property and emit a signal if it has changed."""
        if hasattr(self, property_name):
            if getattr(self, property_name) != value:
                setattr(self, property_name, value)
                self.property_changed.emit(property_name, value)
