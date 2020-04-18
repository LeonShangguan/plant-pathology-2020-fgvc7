from library import *
HEIGHT, WIDTH = 512, 512
MODEL_NAME = ['efficientnet_b7', 'resnet34', 'se_resnext101', 'pnasnet5large']


class Config:
	# Data Split Config
	data_path = "../plant-pathology-2020-fgvc7"
	spilt_save_path = "../output"
	n_split = 5
	split_seed = 960630
	spilt_method = "StratifiedKFold"

	# Training Config
	model_name = MODEL_NAME[1]
	epoches = 30
	batch_size = 32
	num_workers = 16
	accumulate = 4
	train_seed = 42
	model_save_path = "../output"
	oof_save_path = "../output"

	# Data Augmentation Config
	transforms_train = A.Compose([
		# Spatial-level transforms
		# A.RandomResizedCrop(height=HEIGHT, width=WIDTH, p=1.0),
		A.CenterCrop(height=HEIGHT, width=WIDTH, p=1.0),
		# A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
		A.RandomRotate90(p=0.5),
		A.Transpose(p=0.5),
		A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, border_mode=1, rotate_limit=45, p=0.8),

		A.VerticalFlip(p=0.5),
		A.HorizontalFlip(p=0.5),
		A.OneOf([
			A.ElasticTransform(p=1.0),
			A.IAAPiecewiseAffine(p=1.0)
		], p=0.5),
		A.OneOf([
			A.RandomBrightnessContrast(0.15, p=1),
			A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=1),
		], p=0.5),

		A.OneOf([
			A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),
			A.IAASharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0)),
		], p=0.5),

		# A.Lambda(image=pre_trans),
		# ToTensorV2(),
		# A.OneOf([
		# 	A.ElasticTransform(p=1.0),
		# 	A.IAAPiecewiseAffine(p=1.0)
		# ], p=0.5),
		#
		# # Pixel-level transforms
		# A.OneOf([
		# 	A.IAAEmboss(p=1.0),
		# 	A.IAASharpen(p=1.0),
		# 	A.Blur(p=1.0),
		# 	A.MedianBlur(p=1.0),
		# 	A.MotionBlur(p=1.0),
		# ], p=0.5),
		#
		# A.OneOf([
		# 	A.IAAAdditiveGaussianNoise(p=1.0),
		# 	A.RandomGamma(p=1.0),
		# 	A.CLAHE(p=1.0),
		# 	A.RandomBrightnessContrast(p=1.0),
		# ], p=0.5),

		# Normalize, mean & std calculated by EDA. Channel BGR.
		# A.Normalize(p=1),
		A.Normalize(mean=(0.40379888, 0.5128721, 0.31294255), std=(0.20503741, 0.18957737, 0.1883159), p=1.0),
		ToTensorV2(p=1.0),
	])

	transforms_valid = A.Compose([
		# Normalize, mean & std calculated by EDA. Channel BGR.
		A.Resize(height=HEIGHT, width=WIDTH, p=1.0),
		# A.Normalize(p=1),
		# A.Normalize(mean=(0.313, 0.513, 0.404), std=(0.173, 0.176, 0.191), p=1.0),
		A.Normalize(mean=(0.40379888, 0.5128721, 0.31294255), std=(0.20503741, 0.18957737, 0.1883159), p=1.0),
		ToTensorV2(p=1.0),
	])
