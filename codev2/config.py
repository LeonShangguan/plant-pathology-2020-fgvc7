from library import *

MODEL_NAME = ['resnet18', 'resnet34', 'efficientnet_b5', 'efficientnet_b7', 'se_resnext101', 'inceptionv4']


class Config:
    # Data Split Config
    data_path = "../plant-pathology-2020-fgvc7/"
    spilt_save_path = "../output/"
    n_split = 3
    split_seed = 960630
    spilt_method = "StratifiedKFold"

    # Training Config
    model_name = MODEL_NAME[3]
    scheduler_each_iter = False
    learning_rate = 5e-5
    accumulate = 4
    epoches = 20
    batch_size = 4
    num_workers = 8
    train_seed = 42
    model_save_path = "../output/"
    oof_save_path = "../output/"

    #Data augmentation
    transforms_train = A.Compose([
        A.CenterCrop(height=1024, width=1024, p=1.0),
        A.Resize(height=512, width=512, p=1.0),

        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, border_mode=1, rotate_limit=45, p=0.5),

        # Pixels
        A.OneOf([
            A.IAAEmboss(p=0.5),
            A.IAASharpen(p=0.5),
            A.Blur(p=0.5),
            A.ElasticTransform(p=0.5),
        ], p=0.5),

        # Affine
        A.OneOf([
            A.ElasticTransform(p=1.0),
            A.IAAPiecewiseAffine(p=1.0)
        ], p=0.5),

        A.Normalize(mean=(0.404, 0.513, 0.313), std=(0.205, 0.190, 0.188), p=1.0),
        ToTensorV2(p=1.0),
    ])

    transforms_valid = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(mean=(0.404, 0.513, 0.313), std=(0.205, 0.190, 0.188), p=1.0),
        ToTensorV2(p=1.0),
    ])

    transforms_test = A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        A.Normalize(mean=(0.404, 0.513, 0.313), std=(0.205, 0.190, 0.188), p=1.0),
        ToTensorV2(p=1.0),
    ])
