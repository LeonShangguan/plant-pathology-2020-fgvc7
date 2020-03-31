from library import *
SIZE = 512
MODEL_NAME = ['efficientnet_b7', 'resnet34', 'se_resnext101']


class Config:
	# Data Split Config
	data_path = "../plant-pathology-2020-fgvc7"
	spilt_save_path = "../output"
	n_split = 5
	split_seed = 960630
	spilt_method = "StratifiedKFold"

	# Training Config
	model_name = MODEL_NAME[2]
	epoches = 10
	batch_size = 16
	num_workers = 16
	train_seed = 42
	model_save_path = "../output"
	oof_save_path = "../output"

	# Data Augmentation Config
	transforms_train = A.Compose([
		# Spatial-level transforms
		A.RandomResizedCrop(height=SIZE, width=SIZE, p=1.0),
		A.Flip(p=0.8),
		A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.8),

		A.OneOf([
			A.ElasticTransform(p=1.0),
			A.IAAPiecewiseAffine(p=1.0)
		], p=0.5),

		# Pixel-level transforms
		# A.OneOf([
		# 	A.IAAEmboss(p=1.0),
		# 	A.IAASharpen(p=1.0),
		# 	A.Blur(p=1.0),
		# 	A.MedianBlur(p=1.0),
		# 	A.MotionBlur(p=1.0),
		# ], p=0.5),

		# A.OneOf([
		# 	A.IAAAdditiveGaussianNoise(p=1.0),
		# 	A.RandomGamma(p=1.0),
		# 	A.CLAHE(p=1.0),
		# 	A.RandomBrightnessContrast(p=1.0),
		# ], p=0.5),

		# Normalize, mean & std calculated by EDA. Channel BGR.
		A.Normalize(mean=(0.313, 0.513, 0.404), std=(0.173, 0.176, 0.191), always_apply=True),
		ToTensorV2(p=1.0),
	])

	transforms_valid = A.Compose([
		# Normalize, mean & std calculated by EDA. Channel BGR.
		A.Resize(height=SIZE, width=SIZE, p=1.0),
		A.Normalize(mean=(0.313, 0.513, 0.404), std=(0.173, 0.176, 0.191), always_apply=True),
		ToTensorV2(p=1.0),
	])
