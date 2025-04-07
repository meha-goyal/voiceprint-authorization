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
enrolled_speakers = []

def record_audio(filename, duration=DURATION):
    print(f"Recording for {duration} seconds. State wakeword.'")
    recording = sd.rec(int(duration * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, channels=1, 
                       dtype='int16')
    sd.wait()
    wav.write(filename, SAMPLE_RATE, recording)

def verify_speaker(command):
    for speaker in enrolled_speakers:
        score, matched = verification.verify_files(speaker, command)
        if matched:
            print(f"Verification score: {score}")
            return matched
    return False 

def enroll_speaker(num):
    enroll_file = f"enrolled_user_{num}.wav"
    print("Enrolling user ...")
    record_audio(enroll_file)
    print("Enrollment complete.")
    enrolled_speakers.append(enroll_file)

def main():

    print("First, enroll a user.")
    enroll_speaker(0)

    active = input("Would you like to continue? y/n: ")

    while (active != "n"):
        record_audio("command.wav")
        matched = verify_speaker("command.wav")
        
        if matched:
            print("Command Accepted")
            # Execute command
        else:
            print("Command Denied")
            break
        
        active = input("Would you like to continue? Answer with yes or no.")

if __name__ == "__main__":
    main()
