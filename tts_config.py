import os
import zipfile
from typing import NamedTuple, Optional

import gdown
import torch
import yaml
from parallel_wavegan.utils import read_hdf5
from sklearn.preprocessing import StandardScaler

class TTSConfig(NamedTuple):
	model_dir_path: str
	model_path: str
	acoustic_model_path: str
	acoustic_model_config_path: str
	acoustic_model_stats_path: str
	vocoder_model_path: str
	vocoder_stats_path: str
	use_vocoder_stats_flag: bool
	scaler: Optional[StandardScaler]
	device: str
	
	@classmethod
	def get_config_from_path(cls, model_dir_path: str = "./models"):
		"""
		get_config_from_version(cls, model_version: str, download_path: str = './models')
		"""
		acoustic_path = "200epoch.pth"
		vocoder_path = "checkpoint-300000steps.pkl"
		use_vocoder_stats_flag = True
		
		model_path = f"{model_dir_path}/TSUKUYOMICHAN"
		acoustic_model_path = f"{model_path}/ACOUSTIC_MODEL/{acoustic_path}"
		acoustic_model_config_path = f"{model_path}/ACOUSTIC_MODEL/config.yaml"
		acoustic_model_stats_path = f"{model_path}/ACOUSTIC_MODEL/feats_stats.npz"
		vocoder_model_path = f"{model_path}/VOCODER/{vocoder_path}"
		vocoder_stats_path = f"{model_path}/VOCODER/stats.h5"

		scaler = cls.get_scaler(vocoder_stats_path) if use_vocoder_stats_flag else None
		#####################################################################################
		device = "cuda" if torch.cuda.is_available() else "cpu"
		#device = "mps" # for M1mac setting
		#device = torch.device("mps")
		#####################################################################################


		return TTSConfig(model_dir_path=model_dir_path,
						 model_path=model_path,
						 acoustic_model_path=acoustic_model_path,
						 acoustic_model_config_path=acoustic_model_config_path,
						 acoustic_model_stats_path=acoustic_model_stats_path,
						 vocoder_model_path=vocoder_model_path,
						 vocoder_stats_path=vocoder_stats_path,
						 use_vocoder_stats_flag=use_vocoder_stats_flag,
						 scaler=scaler,
						 device=device)

	@staticmethod
	def get_scaler(vocoder_stats_path: str) -> StandardScaler:
		stats = vocoder_stats_path
		scaler = StandardScaler()
		scaler.mean_ = read_hdf5(stats, "mean")
		scaler.scale_ = read_hdf5(stats, "scale")
		scaler.n_features_in_ = scaler.mean_.shape[0]
		return scaler

