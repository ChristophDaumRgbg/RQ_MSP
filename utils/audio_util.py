import numpy as np


# Follows the approach of Schmidt et al. (https://arxiv.org/abs/2501.06117)
def normalize_volume(audio_data, target_db=-25.0):
    audio_data = np.asarray(audio_data, dtype=np.float32)
    rms = np.sqrt(np.mean(audio_data ** 2))  # Calculate the RMS of the waveform
    rms_db = 20 * np.log10(rms + 1e-9)  # Convert RMS to decibels
    gain_db = target_db - rms_db  # Calculate gain needed to reach target
    gain = 10 ** (gain_db / 20)  # Convert gain from dB to linear scale
    return audio_data * gain

def get_noise_category(noise):
    # Fitted to the ESC-50 dataset
    # Partition into Natural and Non-Natural as well as continuous and punctuated
    noise_nature, noise_flow = None, None
    mechanical_sounds = ["vacuum_cleaner", "door_wood_knock", "can_opening", "fireworks", "chainsaw", "airplane", "mouse_click", "train", "church_bells", "clock_alarm", "keyboard_typing", "car_horn", "helicopter", "engine", "hand_saw", "glass_breaking", "toilet_flush", "washing_machine", "clock_tick", "siren", "door_wood_creaks"]
    natural_sounds = ["dog", "chirping_birds", "thunderstorm", "crow", "clapping", "pouring_water", "sheep", "water_drops", "wind", "footsteps", "frog", "cow", "brushing_teeth", "crackling_fire", "drinking_sipping", "rain", "insects", "laughing", "hen", "breathing", "crying_baby", "coughing", "snoring", "pig", "sneezing", "rooster", "sea_waves", "cat", "crickets"]
    continuous_sounds = [ 'chirping_birds', 'vacuum_cleaner', 'thunderstorm', 'clapping', 'chainsaw', 'airplane', 'pouring_water', 'keyboard_typing', 'clock_alarm', 'helicopter', 'church_bells', 'hand_saw', 'train', 'laughing', 'crying_baby', 'crackling_fire', 'wind', 'brushing_teeth', 'rain', 'insects', 'engine', 'breathing', 'snoring', 'washing_machine',  'sea_waves', 'siren', 'door_wood_creaks', 'crickets' ]
    punctuated_sounds = ['dog', 'door_wood_knock', 'can_opening', 'crow', 'fireworks', 'mouse_click', 'sheep', 'water_drops', 'footsteps', 'frog', 'cow', 'car_horn', 'drinking_sipping', 'hen', 'coughing', 'glass_breaking', 'toilet_flush', 'pig', 'clock_tick', 'sneezing', 'rooster', 'cat']
    if noise in mechanical_sounds:
        noise_nature = "Mechanical"
        if noise in continuous_sounds:
            noise_flow = "Continuous"
        elif noise in punctuated_sounds:
            noise_flow = "Punctuated"
    elif noise in natural_sounds:
        noise_nature = "Natural"
        if noise in continuous_sounds:
            noise_flow = "Continuous"
        elif noise in punctuated_sounds:
            noise_flow = "Punctuated"
    else:
        raise ValueError(f"Unknown noise type: {noise}")
    return noise_nature, noise_flow

def get_sound_silence_ratio(audio, sample_rate = 44100, threshold = 0.01, frame_length_ms = 20):
    audio = np.asarray(audio, dtype=np.float32)
    frame_size = int(sample_rate * frame_length_ms / 1000)
    num_frames = (len(audio) + frame_size - 1) // frame_size
    total_samples = num_frames * frame_size

    if len(audio) < total_samples:
        audio = np.pad(audio, (0, total_samples - len(audio)), mode='constant')
    elif len(audio) > total_samples:
        audio = audio[:total_samples]

    # Reshape into frames
    frames = audio.reshape(num_frames, frame_size)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))
    non_silent_frames = np.sum(rms_per_frame > threshold)
    return (non_silent_frames / num_frames) * 100.0

def combine_audio(speech_dict, noise_dict, noise_gain=1, noise_flow="Continuous", target_sr=16000):
    speech = np.asarray(speech_dict["array"], dtype=np.float32)
    noise = np.asarray(noise_dict["array"], dtype=np.float32)

    speech = normalize_volume(speech)
    noise = normalize_volume(noise)

    noise *= noise_gain

    speech_len, noise_len = len(speech), len(noise)

    sound_silence_ratio = get_sound_silence_ratio(noise)

    if noise_flow == "Punctuated" and sound_silence_ratio >= 80:
        if noise_len < speech_len:
            # Generate random start position for the noise within the speech length
            start_position = np.random.randint(0, speech_len - noise_len + 1)
            padded_noise = np.zeros(speech_len, dtype=np.float32)
            padded_noise[start_position:start_position + noise_len] = noise
            noise = padded_noise
    elif noise_flow == "Continuous" or sound_silence_ratio < 80:
        if noise_len < speech_len:
            # Repeat noise until it's at least as long as speech
            repeats = (speech_len // noise_len) + 1
            noise = np.tile(noise, repeats)

    noise = noise[:speech_len]
    combined = np.clip(speech + noise, -1.0, 1.0)

    return combined
