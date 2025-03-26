import os
import torch
import torchaudio
from speechbrain import pretrained
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

verification = pretrained.SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

SAMPLE_RATE = 16000
DURATION = 3  
ENROLL_FILE = "enrolled_user.wav"
KEY_PHRASE = "unlock the button"

def record_audio(filename, duration=DURATION):
    print(f"Recording for {duration} seconds. Speak: '{KEY_PHRASE}'")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, SAMPLE_RATE, recording)

def verify_speaker(enroll_file, test_file):
    score, prediction = verification.verify_files(enroll_file, test_file)
    print(f"Verification score: {score}")
    return prediction 

def main():
    if not os.path.exists(ENROLL_FILE):
        print("Enrolling user...")
        record_audio(ENROLL_FILE)
        print("Enrollment complete. Say the key phrase to unlock.")
    else:
        # Test speaker
        record_audio("test_input.wav")
        matched = verify_speaker(ENROLL_FILE, "test_input.wav")
        if matched:
            print("Access granted. Button unlocked!")
        else:
            print("Access denied. Voice does not match.")

if __name__ == "__main__":
    main()
