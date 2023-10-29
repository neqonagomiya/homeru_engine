# homeyomi by TSUKUYOMICHAN

import numpy as np
import torch
import librosa
#from espnet.bin.tts_inference import Text2Speech
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model

from tts_config import TTSConfig

class HomeyomiEngine:
	def __init__(self, model_path="./models"):
		# def __init__(self, model_version='v.1.2.0'):
		self.config: TTSConfig = TTSConfig.get_config_from_path(model_path)
		self.acoustic_model = self.set_acoustic_model()
		self.vocoder = self.set_vocoder()

	def set_acoustic_model(self):
		acoustic_model = Text2Speech(
			self.config.acoustic_model_config_path,
			self.config.acoustic_model_path,
			device=self.config.device,
			threshold=0.5,
			minlenratio=0.0,
			maxlenratio=10.0,
			use_att_constraint=False,
			backward_window=1,
			forward_window=3
		)
		acoustic_model.spc2wav = None
		return acoustic_model

	def set_vocoder(self):
		vocoder = load_model(self.config.vocoder_model_path).to(self.config.device).eval()
		vocoder.remove_weight_norm()
		return vocoder

	def homeru(self, text, seed=0):
		np.random.seed(seed)
		torch.manual_seed(seed)
		with torch.no_grad():
			_, mel, mel_dnorm, *_ = self.acoustic_model(text)
			if self.config.use_vocoder_stats_flag:
				mel = self.config.scaler.transform(mel_dnorm.cpu())
			wav = self.vocoder.inference(mel)
		wav = wav.view(-1).cpu().numpy()
		# mp3_data
		return wav		
