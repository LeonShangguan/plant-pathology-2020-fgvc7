import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import os
import argparse
from sklearn.model_selection import GroupKFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


############################################ Define augments for test

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--data_path', type=str,
                    default="/media/jionie/my_disk/Kaggle/Plant/input/plant-pathology-2020-fgvc7/",
                    required=False, help='specify the path for data')
parser.add_argument('--n_splits', type=int, default=5, required=False, help='specify the number of folds')
parser.add_argument('--seed', type=int, default=42, required=False,
                    help='specify the random seed for splitting dataset')
parser.add_argument('--save_path', type=str,
                    default="/media/jionie/my_disk/Kaggle/Plant/input/plant-pathology-2020-fgvc7/", required=False,
                    help='specify the path for saving splitted csv')
parser.add_argument('--fold', type=int, default=0, required=False,
                    help='specify the fold for testing dataloader')
parser.add_argument('--batch_size', type=int, default=4, required=False,
                    help='specify the batch_size for testing dataloader')
parser.add_argument('--val_batch_size', type=int, default=4, required=False,
                    help='specify the val_batch_size for testing dataloader')
parser.add_argument('--num_workers', type=int, default=0, required=False,
                    help='specify the num_workers for testing dataloader')
parser.add_argument('--split', type=str,
                    default="StratifiedKFold", required=False,
                    help='specify how we split csv')


class PlantDataset(Dataset):

    def __init__(self,
                 df,
                 data_path='../plant-pathology-2020-fgvc7',
                 transforms=None):
        self.df = df
        self.transforms = transforms
        self.data_path = data_path
        self.labels = self.df[['healthy', 'multiple_diseases', 'rust', 'scab']].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = os.path.join(self.data_path, 'images/{}.jpg'.format(self.df.loc[idx, 'image_id']))
        image = cv2.imread(image_src)
        labels = self.labels[idx]
        labels = torch.from_numpy(labels.astype(np.int8))

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, labels


############################################ Define getting data split functions
def get_train_val_split(data_path="/media/jionie/my_disk/Kaggle/Plant/input/plant-pathology-2020-fgvc7/train.csv",
                        save_path="/media/jionie/my_disk/Kaggle/Plant/input/plant-pathology-2020-fgvc7/",
                        n_splits=5,
                        seed=960630,
                        split="StratifiedKFold"):

    df_path = os.path.join(data_path, "train.csv")
    os.makedirs(os.path.join(save_path, 'split/{}'.format(split)), exist_ok=True)
    df = pd.read_csv(df_path, encoding='utf8')
    df['label'] = 0*df['healthy'] + 1*df['multiple_diseases']+2*df['rust']+3*df['scab']

    if split == "MultilabelStratifiedKFold":
        kf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=seed).split(df,
                                                                                   df[['healthy',
                                                                                       'multiple_diseases',
                                                                                       'rust',
                                                                                       'scab']].values)
    elif split == "StratifiedKFold":
        kf = StratifiedKFold(n_splits=n_splits, random_state=seed).split(df,
                                                                         df[['label']].values)
    elif split == "GroupKFold":
        kf = GroupKFold(n_splits=n_splits).split(df, groups=df[
            ['healthy', 'multiple_diseases', 'rust', 'scab']].values)
    else:
        raise NotImplementedError

    for fold, (train_idx, valid_idx) in enumerate(kf):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[valid_idx]

        df_train.to_csv(os.path.join(save_path, 'split/{}/train_fold_{}_seed_{}.csv'.format(split, fold, seed)))
        df_val.to_csv(os.path.join(save_path, 'split/{}/val_fold_{}_seed_{}.csv'.format(split, fold, seed)))

    return


############################################ Define test_train_val_split functions
def test_train_val_split(data_path,
                         save_path,
                         n_splits,
                         seed,
                         split):
    print("------------------------testing train test splitting----------------------")
    print("data_path: ", data_path)
    print("save_path: ", save_path)
    print("n_splits: ", n_splits)
    print("seed: ", seed)

    get_train_val_split(data_path=data_path, save_path=save_path, n_splits=n_splits, seed=seed, split=split)

    print("generating successfully, please check results !")

    return


############################################ Define get_test_loader functions
def get_test_loader(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                    batch_size=4,
                    num_workers=4,
                    transforms=None):

    df_path = os.path.join(data_path, "sample_submission.csv")
    test_df = pd.read_csv(df_path)

    test_dataset = PlantDataset(df=test_df, data_path=data_path, transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             drop_last=False)

    return test_loader


############################################ Define get_train_val_loader functions
def get_train_val_loader(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                         fold=0,
                         seed=960630,
                         split="StratifiedKFold",
                         batch_size=4,
                         val_batch_size=4,
                         num_workers=4,
                         transforms=None,
                         val_transforms=None):

    train_df_path = os.path.join(data_path, 'split/{}/train_fold_{}_seed_{}.csv'.format(split, fold, seed))
    val_df_path = os.path.join(data_path, 'split/{}/val_fold_{}_seed_{}.csv'.format(split, fold, seed))

    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)

    train_dataset = PlantDataset(df=train_df, data_path=data_path, transforms=transforms)
    val_dataset = PlantDataset(df=val_df, data_path=data_path, transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers, shuffle=False,
                            drop_last=False)

    return train_loader, val_loader


############################################ Define test_test_loader functions
def test_test_loader(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                     batch_size=4,
                     num_workers=4,
                     transforms=None):

    test_loader = get_test_loader(data_path=data_path, batch_size=batch_size, num_workers=num_workers,
                                  transforms=transforms)

    for i, (image, label) in enumerate(test_loader):
        print("----------------------test test_loader-------------------")
        print("image shape: ", image.shape)
        print("label shape: ", label.shape)
        print("-----------------------finish testing------------------------")

        break

    return


############################################ Define test_train_val_loader functions
def test_train_val_loader(data_path="/media/jionie/my_disk/Kaggle/Tweet/input/tweet-sentiment-extraction/",
                          fold=0,
                          seed=960630,
                          split="StratifiedKFold",
                          batch_size=4,
                          val_batch_size=4,
                          num_workers=4,
                          transforms=None,
                          val_transforms=None):

    train_loader, val_loader = get_train_val_loader(data_path=data_path, fold=fold, seed=seed, split=split,
                                                    batch_size=batch_size, val_batch_size=val_batch_size,
                                                    num_workers=num_workers, transforms=transforms,
                                                    val_transforms=val_transforms)

    for i, (image, label) in enumerate(train_loader):
        print("----------------------test train_loader-------------------")
        print("image shape: ", image.shape)
        print("label shape: ", label.shape)
        print("-----------------------finish testing------------------------")

        break

    for i, (image, label) in enumerate(val_loader):
        print("----------------------test val_loader-------------------")
        print("image shape: ", image.shape)
        print("label shape: ", label.shape)
        print("-----------------------finish testing------------------------")

        break

    return


if __name__ == '__main__':

    args = parser.parse_args()

    test_train_val_split(data_path=args.data_path,
                         save_path=args.save_path,
                         n_splits=args.n_splits,
                         seed=args.seed,
                         split=args.split)

    test_test_loader(data_path=args.data_path,
                     batch_size=args.batch_size,
                     num_workers=args.num_workers,
                     transforms=None)

    test_train_val_loader(data_path=args.data_path,
                          fold=args.fold,
                          seed=args.seed,
                          split=args.split,
                          batch_size=args.batch_size,
                          val_batch_size=args.val_batch_size,
                          num_workers=args.num_workers,
                          transforms=None,
                          val_transforms=None)