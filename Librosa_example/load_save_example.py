"""
Author: songjun
Date: 2018/5/5
Description:
Usage:
"""
import librosa

file_path = "E:\\Datasets\\audio_example\\wav_example.wav"

y, sr = librosa.load(file_path, sr=11025, mono=True, offset=30, duration=60)

librosa.output.write_wav("example.wav", y)