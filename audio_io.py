import threading
import sounddevice as sd
import numpy as np
import soundfile as sf
import time
import os
from settings import STREAMMING_INPUT_BLOCKSIZE
import settings
from scipy.signal import resample


class AudioIO:
    """Real-time audio I/O: captures microphone and feeds model buffers."""
    def __init__(self, model, samplerate=settings.SAMPLE_RATE, blocksize=STREAMMING_INPUT_BLOCKSIZE):
        self._eq_model = model # Reference to the EQModel
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = settings.AUDIO_CHANNELS
        self.stream = None
        self.running = False
        self._input_source = "system"  # Changed from "mic" to "system"
        self._streaming = False
        self._file_path = None
        self._file_thread = None
        self._file_data = None
        self._file_position = 0

        # List all audio devices at startup
        self._list_all_audio_devices()
        
        # Add locks for thread safety
        self._buffer_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._file_lock = threading.Lock()
        
        # Create DSP engine for processing
        from backend.dsp_engine import DSPEngine
        self.dsp_engine = DSPEngine(model)
        
        # Initialize output device once
        self.output_device = self._init_plaback_device()
        
        # Initialize system audio by default
        self._init_system_audio()

    def _init_plaback_device(self):
        """Initialize and return the output device based on settings.PLAYBACK_DEVICE_NAME."""
       
        # # Debug: list all devices what have 2 output channels
        # print("\nAvailable output devices with 2 channels:")
        # for idx, dev in enumerate(sd.query_devices()):
        #     if dev['max_output_channels'] == 2:
        #         print(f"{idx}: {dev['name']} (max_output_channels: {dev['max_output_channels']}, default_samplerate: {dev['default_samplerate']})")
       
        output_device = None
        
        # Find output device with specified name
        for idx, dev in enumerate(sd.query_devices()):
            if (settings.PLAYBACK_DEVICE_NAME in dev['name'] and 
                dev['max_output_channels'] == 2 
                and dev['default_samplerate'] == self.samplerate
                ):
                output_device = idx
                print(f"\nThe specified playback output device is found: {dev['name']} (index: {idx}, max_output_channels: {dev['max_output_channels']}, default_samplerate: {dev['default_samplerate']})")
                break
        
        # If specified output not found, use default
        if output_device is None:
            raise RuntimeError(f"Specified playback output device {settings.PLAYBACK_DEVICE_NAME} (channel_num: {self.channels}, samplerate: {self.samplerate}) not found.")

        return output_device

    def _mic_callback(self, indata, frames, time, status):
        if status:
            print(status)
        # Flatten and convert to list for model
        # Preserve 2D structure even for mono
        buffer = indata[:, 0:1].copy()  # mono as (n_samples, 1)
        
        # Process audio through DSP engine
        processed = self.dsp_engine.process_audio(buffer)
        
        # Update input and output signals with lock protection
        with self._buffer_lock:
            # Update input signal in model
            self._eq_model.input_signal = buffer.tolist()
            # Update output signal in model
            self._eq_model.output_signal = processed

    def set_input_source(self, source):
        """Set the input source: 'system', 'mic', or 'file'."""
        # Always stop streaming, even if not running
        self.stop_streaming()
        self._input_source = source
        print(f"Input source set to: {source}")
        self.dsp_engine.reset_filter_states()
        if source == "system":
            self._init_system_audio()
        elif source == "mic":
            self._init_microphone()
        elif source == "file":
            pass  # File is initialized separately via load_file
        # Optionally, update model/UI state here

    def set_input_file(self, file_path):
        """
        Set the input file path for file source.

        Args:
            file_path (str): Path to the audio file
        """
        self._file_path = file_path
        print(f"Input file set to: {file_path}")

        # Raise an error if the file does not exist
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # If file source is selected, load the file
        if self._input_source == "file":
            self._load_file(self._file_path)

    def _load_file(self, file_path):
        """Load and process an audio file."""
        try:
            # Stop any existing file playback
            if self._file_thread and self._file_thread.is_alive():
                self.stop()
            
            # Load the audio file
            data, sample_rate = sf.read(file_path, always_2d=True)
            
            # Convert to mono 
            if settings.CONVERT_TO_MONO:
                data = stereo_to_mono(data)

            # Resample if necessary
            if sample_rate != self.samplerate:
                print(f"Resampling from {sample_rate} Hz to {self.samplerate} Hz...")
                num_samples = int(data.shape[0] * self.samplerate / sample_rate)
                # Resample each channel separately if stereo
                if data.ndim == 2 and data.shape[1] > 1:
                    data_resampled = np.zeros((num_samples, data.shape[1]), dtype=data.dtype)
                    for ch in range(data.shape[1]):
                        data_resampled[:, ch] = resample(data[:, ch], num_samples)
                    data = data_resampled
                else:
                    data = resample(data, num_samples)
                sample_rate = self.samplerate  # Update to new sample rate`

                
            # Store the file data
            self._file_data = data
            self._file_position = 0
            
            print(f"Loaded file: {len(data)} samples")
            
            # Start offline processing
            self.start()
            
        except Exception as e:
            print(f"Error loading audio file: {e}")
            self._file_data = None

    def start(self):
        """Start audio processing based on input source."""
        with self._state_lock:
            if self.running:
                return
            self.running = True
            self._streaming = True

        with self._file_lock:
            is_file = self._input_source == "file" and self._file_data is not None

        if is_file:
            # Start a thread for file playback simulation
            self._file_thread = threading.Thread(target=self._file_playback)
            self._file_thread.daemon = True
            self._file_thread.start()

        elif self._input_source == "system" or self._input_source == "mic":
            # Use duplex stream for both system audio and mic
            input_device, output_device = self.input_device, self.output_device
            
            # # Debug: print input and output device info
            # device_list = sd.query_devices()
            # input_info = device_list[input_device] if input_device is not None else None
            # output_info = device_list[output_device] if output_device is not None else None
            # print(f"Selected input device: {input_info['name']} (index: {input_device}, max_input_channels: {input_info['max_input_channels']})")
            # print(f"Selected output device: {output_info['name']} (index: {output_device}, max_output_channels: {output_info['max_output_channels']})")
            # print(f"Starting duplex stream with input device {input_device} and output device {output_device}")

            # For mic input, use (1, 2) channels - mono in, stereo out
            # For system input, use existing channels setting
            channels = (1, 2) if self._input_source == "mic" else self.channels
            min_samplerate = min(sd.query_devices(input_device, 'input')['default_samplerate'], sd.query_devices(output_device, 'output')['default_samplerate'])
            
            # device_list = sd.query_devices()
            # input_info = device_list[input_device] if input_device is not None else None
            # output_info = device_list[output_device] if output_device is not None else None
            # print(f"Selected input device: {input_info['name']} (index: {input_device}, max_input_channels: {input_info['max_input_channels']})")
            # print(f"Selected output device: {output_info['name']} (index: {output_device}, max_output_channels: {output_info['max_output_channels']})")

            self.stream = sd.Stream(
                samplerate=min_samplerate,
                blocksize=self.blocksize,
                dtype='float32',
                channels=channels,
                callback=self._duplex_audio_callback,
                device=(input_device, output_device)
            )
            self.stream.start()
        else:
            # Real-time streaming from mic
            self.stream = sd.InputStream(
                channels=1, # Mono input for microphone
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                callback=self._mic_callback
            )
            self.stream.start()
        
        self._eq_model.set_property("is_streaming", True)

    def stop(self):
        """Stop all audio processing."""
        with self._state_lock:
            self.running = False
            self._streaming = False
            
            # Get local reference to stream
            stream = self.stream
            self.stream = None
        
        # Stop streaming if active
        if stream:
            stream.stop()
            stream.close()
        
        # Wait for file thread to finish if active
        file_thread = self._file_thread
        if file_thread and file_thread.is_alive():
            # Give it a short time to terminate naturally
            file_thread.join(0.5)
            # Reset the thread reference
            with self._state_lock:
                self._file_thread = None
            print("File playback thread stopped")

        self._eq_model.set_property("is_streaming", False)



    def stop_streaming(self):
        """Stop the audio streaming completely."""
        with self._state_lock:
            self._streaming = False
            self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        # Stop file thread if running
        if hasattr(self, '_file_thread') and self._file_thread is not None:
            self._file_thread = None
        # Notify model/UI via EQModel signal
        if hasattr(self._eq_model, 'set_property'):
            self._eq_model.set_property("is_streaming", False)
        print("Streaming stopped")

    def is_file_playing(self):
        """Check if a file is currently being played back."""
        return (self._input_source == "file" and 
                self.running and 
                self._streaming and
                self._file_data is not None and
                self._file_position < len(self._file_data))

    def get_playback_position(self):
        """Get current playback position in seconds."""
        with self._file_lock:
            if self._file_data is not None and self.samplerate > 0:
                return self._file_position / self.samplerate
        return 0.0

    def get_file_duration(self):
        """Get total file duration in seconds."""
        with self._file_lock:
            if self._file_data is not None and self.samplerate > 0:
                return len(self._file_data) / self.samplerate
        return 0.0

    def is_mic_active(self):
        """Check if microphone is active."""
        return self._input_source == "mic" and self.running and self._streaming

    def get_input_buffer(self):
        """Get the current audio input buffer."""
        with self._buffer_lock:
            return np.array(self._eq_model.input_signal) if self._eq_model.input_signal else np.zeros(0)

    def get_output_buffer(self):
        """Get the current audio output buffer."""
        with self._buffer_lock:
            return np.array(self._eq_model.output_signal) if self._eq_model.output_signal else np.zeros(0)

    def _init_microphone(self):
        """Initialize microphone capture with playback capability."""
        self._eq_model.input_signal = []
        self._eq_model.output_signal = []
        
        try:
            # Find microphone device with specified name
            # self.input_device = sd.default.device[0]  # Default input device, ie. mic

            for idx, dev in enumerate(sd.query_devices()):
                if (settings.MIC_DEVICE_NAME in dev['name'] and 
                    dev['max_input_channels'] == 2 # assume mic is stereo - maybe need more code to handle the case of mono
                    and dev['default_samplerate'] == self.samplerate):  
                    self.input_device = idx
                    print(f"Found microphone input device: {dev['name']} (index: {idx}, max_input_channels: {dev['max_input_channels']})")
                    break
            
            if self.input_device is None:
                # Instead of using default device, raise an exception
                raise RuntimeError(f"Microphone device '{settings.MIC_DEVICE_NAME}' with "
                                   f"2 channels and {self.samplerate} not found. Please check your settings.")

            # if self.input_device is None:
            #     # Use default microphone device instead of raising an exception
            #     self.input_device = sd.default.device[0]  # Default input device
            #     device_info = sd.query_devices()[self.input_device]
            #     print(f"Specified microphone device '{settings.MIC_DEVICE_NAME}' not found. Using default: {device_info['name']} (index: {self.input_device})")
                
            # Get device details
            device_list = sd.query_devices()
            input_info = device_list[self.input_device]
            output_info = device_list[self.output_device]

            print(f"Microphone initialized: Input = {input_info['name']} (index {self.input_device}, samplerate {input_info['default_samplerate']}, channels {input_info['max_input_channels']}), "
                  f"Output = {output_info['name']} (index {self.output_device}, samplerate {output_info['default_samplerate']}, channels {output_info['max_output_channels']})")
              
        except Exception as e:
            print(f"Error initializing microphone: {e}")

    def _init_system_audio(self):
        """Initialize system audio output capture using VB-CABLE."""
        self._eq_model.input_signal = []
        self._eq_model.output_signal = []

        try:

            self.input_device = self._find_vb_audio_virtual_cable()
            
            if self.input_device is None:
                print("Required audio input device 'VB-Audio Virtual Cable' not found. Available devices:")
                # for idx, dev in enumerate(sd.query_devices()):
                #     print(f"{idx}: {dev['name']} (In: {dev['max_input_channels']}, Out: {dev['max_output_channels']})")
                raise RuntimeError("VB-Audio Virtual Cable device not found.")

            # Get device details
            device_list = sd.query_devices()
            input_info = device_list[self.input_device]
            output_info = device_list[self.output_device]
            
            # Use default sample rate of the device
            # self.samplerate = int(input_info['default_samplerate'])
            # input_channels = input_info['max_input_channels']
            # output_channels = output_info['max_output_channels']

            print(f"\nSystem audio initialized: Input = {input_info['name']} ({self.input_device}), "
                  f"Output (Playback) = {output_info['name']} ({self.output_device}), "
                  f"Channels={self.channels}, Samplerate={self.samplerate}")
    
        except Exception as e:
            print(f"Error initializing system audio: {e}")

    def _find_vb_audio_virtual_cable(self):
        """Find the appropriate input device (CABLE Output)."""

        # Debug: list all devices what have 2 input or output channels
        # print("\nAvailable input devices with 2 channels:")
        # for idx, dev in enumerate(sd.query_devices()):
        #     if dev['max_input_channels'] == 2:
        #         print(f"{idx}: {dev['name']} (max_input_channels: {dev['max_input_channels']}, default_samplerate: {dev['default_samplerate']})")
        # print("")

        input_device = None
        
        for idx, dev in enumerate(sd.query_devices()):
            if ("CABLE Output" in dev['name'] and dev['max_input_channels'] == 2):
                print(f"\nThe specified input device is found: {dev['name']} (index: {idx}, max_input_channels: {dev['max_input_channels']}, default_samplerate: {dev['default_samplerate']})")
                input_device = idx
                break
                
        return input_device

    def _duplex_audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"_duplex_audio_callback status: {status}")
    
        is_stereo = indata.shape[1] > 1 if indata.ndim > 1 else False

        # debug
        # print(f"_duplex_audio_callback Input shape: {indata.shape}")
        #  
        if is_stereo and settings.CONVERT_TO_MONO:
            indata = stereo_to_mono(indata)
            is_stereo =False

        if is_stereo:  # If stereo
            # Create output buffer with same shape as outdata
            processed_stereo = np.zeros_like(outdata)
            
            # Process left channel
            left_channel = indata[:, 0].copy()
            processed_left = self.dsp_engine.process_audio(left_channel)
            # processed_left = np.array(processed_left) # This conversion between Python lists and NumPy arrays can introduce precision issues
            processed_stereo[:, 0] = processed_left

            # processed_stereo[:, 0] =  indata[:, 0].copy() # debug
            
            # Reset filter states before processing right channel
            self.dsp_engine.reset_filter_states()
            
            # Process right channel
            right_channel = indata[:, 1].copy()
            processed_right = self.dsp_engine.process_audio(right_channel)
            # processed_right = np.array(processed_right)
            processed_stereo[:, 1] = processed_right

            # processed_stereo[:, 1] = indata[:, 1].copy() # debug
            
            # Set the output
            outdata[:] = processed_stereo
            
            # Combine stereo channels for plotting (simple average)
            input_mono = stereo_to_mono(indata)
            output_mono = stereo_to_mono(processed_stereo)
            
            # Update model buffers with combined mono signal
            with self._buffer_lock:
                self._eq_model.input_signal = input_mono.tolist()
                self._eq_model.output_signal = output_mono.tolist()
        else:
            # Mono processing
            buffer = indata.copy()
            processed = self.dsp_engine.process_audio(buffer)
            processed = np.array(processed)
            
            # Ensure processed is the correct shape for outdata
            if processed.ndim == 1 and outdata.shape[1] > 1:
                processed = np.tile(processed[:, None], (1, outdata.shape[1]))
            
            outdata[:] = processed
            
            # Update model buffers for plotting 
            with self._buffer_lock:
                # Preserve channel information
                if buffer.ndim > 1:
                    self._eq_model.input_signal = buffer[:, 0].tolist()
                else:
                    # Reshape 1D to 2D before extracting channel
                    self._eq_model.input_signal = buffer.reshape(-1, 1)[:, 0].tolist()

                if processed.ndim > 1:
                    self._eq_model.output_signal = processed[:, 0].tolist() 
                else:
                    # Reshape 1D to 2D before extracting channel
                    self._eq_model.output_signal = processed.reshape(-1, 1)[:, 0].tolist()

    def _file_playback(self):
        """Simulate playback of an audio file in a separate thread."""
        try:
            # Calculate block size for file playback
            # file_blocksize = min(self.blocksize, 1024)  # Limit to 1024 samples per block for file playback

            file_blocksize = self.blocksize

            # Preload a portion of the file into the buffer
            buffer = self._file_data[self._file_position:self._file_position + file_blocksize]
            
            channel_count = 1
            if settings.CONVERT_TO_MONO:
                channel_count = 1
            else:
                channel_count = 2 if buffer.ndim > 1 and buffer.shape[1] == 2 else 1

            # Create output stream
            output_stream = sd.OutputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                device=self.output_device,
                channels=channel_count,
                dtype='float32'
            )
            output_stream.start()

            while self.running and self._streaming and self._file_position < len(self._file_data):
                # Process audio through DSP engine
                processed = self.dsp_engine.process_audio(buffer)
                processed = np.array(processed)

                # # Ensure processed is stereo for output
                # if processed.ndim == 1:
                #     processed = np.column_stack([processed, processed])
                # elif processed.ndim == 2 and processed.shape[1] == 1:
                #     processed = np.column_stack([processed[:, 0], processed[:, 0]])
                # elif processed.ndim == 2 and processed.shape[1] == 2:
                #     pass  # already stereo
                # else:
                #     raise ValueError(f"Unexpected processed shape: {processed.shape}")

                # print(f"Processed shape before write: {processed.shape}")

                # Ensure dtype is float32 for sounddevice
                processed = processed.astype(np.float32)

                # Output processed audio to the stream
                output_stream.write(processed)

                 # update model for plotting 
                with self._buffer_lock:
                    # For mono, ensure shape is (N, 1)
                    if buffer.ndim == 1:
                        input_for_plot = buffer.reshape(-1, 1)
                    else:
                        input_for_plot = buffer
                    if processed.ndim == 1:
                        output_for_plot = processed.reshape(-1, 1)
                    else:
                        output_for_plot = processed

                    # If stereo, plot only the first channel or average
                    self._eq_model.input_signal = input_for_plot[:, 0].tolist()
                    self._eq_model.output_signal = output_for_plot[:, 0].tolist()


                # Update file position
                with self._file_lock:
                    self._file_position += file_blocksize
                    buffer = self._file_data[self._file_position:self._file_position + file_blocksize]
                    # If we've reached the end of the file, loop back to the start
                    if self._file_position >= len(self._file_data):
                        self._file_position = 0
                        self._streaming = False
                        self.stop()
                        break
            
            output_stream.stop()
            output_stream.close()

            # Signal that playback has finished
            self._streaming = False
            self._eq_model.set_property("is_streaming", False)
        
        except Exception as e:
            self._streaming = False
            self._eq_model.set_property("is_streaming", False)
            print(f"Error in file playback: {e}")

    def _list_all_audio_devices(self):
        """List all available input and output devices with settings.AUDIO_CHANNELS channels"""

        print(f"\nIn settings.py, the Microphone device name specified: '{settings.MIC_DEVICE_NAME}', "
              f"and the Playback device name specified: '{settings.PLAYBACK_DEVICE_NAME}'.\n"
              f"Also make sure you have 'VB-Audio Virtual Cable' software installed to allow audio routing for system sound output as an input source.")

        print(f"\nAvailable INPUT devices with {settings.AUDIO_CHANNELS} channels:")
        input_devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] == settings.AUDIO_CHANNELS:
                input_devices.append((idx, dev['name'], dev['default_samplerate']))
                print(f"  {idx}: {dev['name']} (max_input_channels: {dev['max_input_channels']}, default_samplerate: {dev['default_samplerate']})")
        
        if not input_devices:
            print(f"  No input devices with {settings.AUDIO_CHANNELS} channels found")
        
        print("\nAvailable OUTPUT devices with 2 channels:")
        output_devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_output_channels'] == settings.AUDIO_CHANNELS:
                output_devices.append((idx, dev['name'], dev['default_samplerate']))
                print(f"  {idx}: {dev['name']} (max_output_channels: {dev['max_output_channels']}, default_samplerate: {dev['default_samplerate']})")
        
        if not output_devices:
            print(f"  No output devices with {settings.AUDIO_CHANNELS} channels found")
        
        # print("\nDefault Devices:")
        # print(f"  Default Input: {sd.default.device[0]}")
        # print(f"  Default Output: {sd.default.device[1]}")
        print("")

# def find_input_device():
#     input_device = None
    
#     # Debug: list all devices what have 2 input or output channels
#     print("\nAvailable input devices with 2 channels:")
#     for idx, dev in enumerate(sd.query_devices()):
#         if dev['max_input_channels'] == 2:
#             print(f"{idx}: {dev['name']} (max_input_channels: {dev['max_input_channels']}, default_samplerate: {dev['default_samplerate']})")
#     print("")

#     for idx, dev in enumerate(sd.query_devices()):
#         # Find "CABLE Output" with 2 input channels
#         if (input_device is None and "CABLE Output" in dev['name'] and
#             dev['max_input_channels'] == 2):
#             print(f"Found input device: {dev['name']} (index: {idx}, max_input_channels: {dev['max_input_channels']})")
#             input_device = idx
#             break

#     # Use the already initialized output_device from AudioIO
#     return input_device  # Return None for output_device as it's initialized in AudioIO.__init__


def stereo_to_mono(audio_data):
    """
    Convert stereo audio data to mono by averaging channels.
    Works with both 2D stereo arrays and already-mono arrays.
    
    Args:
        audio_data (numpy.ndarray): Input audio data (can be stereo or mono)
        
    Returns:
        numpy.ndarray: Mono audio data
    """
    # Check if input is already 1D (mono)
    if audio_data.ndim == 1:
        return audio_data
        
    # Check if input is stereo (has multiple channels)
    if audio_data.shape[1] > 1:
        # Average all channels to create mono
        return np.mean(audio_data, axis=1).reshape(-1, 1)
    
    # Input is 2D but with only one channel
    return np.mean(audio_data, axis=1)  # Use mean instead of flatten for consistency