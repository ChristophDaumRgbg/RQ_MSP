import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Model
import config



# Load Brouhaha model
def load_brouhaha():
    print("loading brouhaha")
    model = Model.from_pretrained("pyannote/brouhaha", use_auth_token=os.getenv("HF_AUTH_TOKEN"))   # Huggingface token is set as environment variable
    model.to(torch.device("cuda"))
    from pyannote.audio import Inference
    inference = Inference(model)
    return inference

# Load Whisper model
def load_whisper():
    print("loading whisper")
    processor = WhisperProcessor.from_pretrained(config.asr_model)
    model = WhisperForConditionalGeneration.from_pretrained(config.asr_model)
    model.config.forced_decoder_ids = None
    model.to(torch.device(config.gpu_device))
    return processor, model

def get_snr(inference, path):
    output = inference(path)
    frame_counter = 0
    snr_sum = 0
    vad_sum = 0
    vad_sum_active = 0
    for frame, (vad, snr, c50) in output:
        snr_sum = snr_sum + snr
        vad_sum = vad_sum + vad
        if vad >= 0.95:
          vad_sum_active = vad_sum_active + 1
        frame_counter = frame_counter + 1
    avg_snr = snr_sum / frame_counter
    avg_vad = vad_sum / frame_counter
    avg_vad_active = vad_sum_active / frame_counter
    return avg_snr, avg_vad, avg_vad_active
