# generate home.wav
import numpy as np
import simpleaudio as sa
import soundfile as sf

#from tts_engine import HomeyomiEngine
from homeyomi_engine import  HomeyomiEngine

# config
output_path = "./output"
MAX_WAV_VALUE = 32768.0
fs = 24000
wav = []
#seed = 0
text = "こんにちは"
voice = ""
homeyomi_engine = HomeyomiEngine()

def make_home_voice(text, wav):
	wav = homeyomi_engine.homeru(text)
	wav = wav * MAX_WAV_VALUE
	wav = wav.astype(np.int16)
	return wav	

def save_voice(output_path, text, wav, fs):
	sf.write(f"{output_path}/homeyomi.wav", wav, fs, "PCM_16")


wav = make_home_voice(text, wav)
save_voice(output_path=output_path, text=text, wav=wav, fs=fs)
