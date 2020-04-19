import os
from utils.augmentation import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

def pre_trans(x, cols, rows):
    return (x * 2.0 - 1.0)

class Config:
    # config settings
    def __init__(self, fold, model_type="Resnet34", seed=2020, batch_size=16, accumulation_steps=1):
        # setting
        self.reuse_model = True
        self.load_from_load_from_data_parallel = False
        self.data_parallel = False  # enable data parallel training
        self.adversarial = False # enable adversarial training, not support now
        self.apex = True  # enable mix precision training
        self.load_optimizer = False
        self.skip_layers = []
        # model
        self.model_type = model_type
        # path, specify the path for data
        self.data_path = '/media/jionie/my_disk/Kaggle/Plant/input/plant-pathology-2020-fgvc7/'
        # path, specify the path for saving splitted csv
        self.save_path = '/media/jionie/my_disk/Kaggle/Plant/input/plant-pathology-2020-fgvc7/'
        # k fold setting
        self.split = "StratifiedKFold"
        self.seed = seed
        self.n_splits = 5
        self.fold = fold
        # path, specify the path for saving model
        self.model_folder = os.path.join("/media/jionie/my_disk/Kaggle/Plant/model", self.model_type)
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
        self.checkpoint_folder_all_fold = os.path.join(self.model_folder, 'seed_' + str(self.seed))
        if not os.path.exists(self.checkpoint_folder_all_fold):
            os.mkdir(self.checkpoint_folder_all_fold)
        self.checkpoint_folder = os.path.join(self.checkpoint_folder_all_fold,'fold_' + str(self.fold) + '/')
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        self.save_point = os.path.join(self.checkpoint_folder, '{}_step_{}_epoch.pth')
        self.load_points = [p for p in os.listdir(self.checkpoint_folder) if p.endswith('.pth')]
        if len(self.load_points) != 0:
            self.load_point = sorted(self.load_points, key=lambda x: int(x.split('_')[0]))[-1]
            self.load_point = os.path.join(self.checkpoint_folder, self.load_point)
        else:
            self.reuse_model = False
        # optimizer
        self.optimizer_name = "AdamW"
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 2
        # lr scheduler, can choose to use proportion or steps
        self.lr_scheduler_name = 'WarmupCosineAnealing'
        self.warmup_proportion = 0
        self.warmup_steps = 0
        # lr
        self.lr = 5e-5
        self.weight_decay = 0.0001
        self.min_lr = 5e-5
        # dataloader settings
        self.batch_size = batch_size
        self.val_batch_size = 32
        self.num_workers = 4
        self.shuffle = True
        self.drop_last = True
        # gradient accumulation
        self.accumulation_steps = accumulation_steps
        # epochs
        self.num_epoch = 30
        # saving rate
        self.saving_rate = 1
        # early stopping
        self.early_stopping = 7 / self.saving_rate
        # progress rate
        self.progress_rate = 1/10
        # transform
        self.HEIGHT = 512
        self.WIDTH = 512
        self.transforms = A.Compose([
            # Spatial-level transforms
            A.OneOf([
                A.RandomResizedCrop(height=self.HEIGHT, width=self.WIDTH, p=1.0),
                A.CenterCrop(height=self.HEIGHT, width=self.WIDTH, p=1.0),
                A.Resize(height=self.HEIGHT, width=self.WIDTH, p=1.0),
            ], p=1),

            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, border_mode=1, rotate_limit=45, p=0.8),

            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),

            A.OneOf([
                A.ElasticTransform(p=1.0),
                A.IAAPiecewiseAffine(p=1.0),
                GridMask(num_grid=(3, 7), p=1),
            ], p=0.5),

            A.OneOf([
                A.RandomBrightnessContrast(0.15, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=1),
            ], p=0.5),

            A.OneOf([
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3)),
                A.IAASharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0)),
            ], p=0.5),

            # A.Normalize(mean=(0.40379888, 0.5128721, 0.31294255), std=(0.20503741, 0.18957737, 0.1883159), p=1.0),
            A.Lambda(image=pre_trans),
            ToTensorV2(p=1.0),
        ])

        self.val_transforms = A.Compose([
            # Normalize, mean & std calculated by EDA. Channel BGR.
            A.Resize(height=self.HEIGHT, width=self.WIDTH, p=1.0),
            # A.Normalize(mean=(0.40379888, 0.5128721, 0.31294255), std=(0.20503741, 0.18957737, 0.1883159), p=1.0),
            A.Lambda(image=pre_trans),
            ToTensorV2(p=1.0),
        ])