from library import *


class PlantDataset(Dataset):

    def __init__(self,
                 df,
                 data_path='../plant-pathology-2020-fgvc7',
                 transform=None):
        self.df = df
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = self.data_path + '/images/' + self.df.loc[idx, 'image_id'] + '.jpg'
        image = cv2.imread(image_src)

        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.unsqueeze(-1)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, labels


def get_train_val_split(df_path="../plant-pathology-2020-fgvc7/train.csv",
                        save_path="../output_v2/",
                        n_splits=5,
                        seed=960630,
                        split="StratifiedKFold"):
    os.makedirs(save_path + '/split/' + split, exist_ok=True)
    df = pd.read_csv(df_path, encoding='utf8')
    df['label'] = 0*df['healthy'] + 1*df['multiple_diseases']+2*df['rust']+3*df['scab']

    kf = StratifiedKFold(n_splits=n_splits, random_state=seed).split(df, df[['label']].values)

    for fold, (train_idx, valid_idx) in enumerate(kf):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[valid_idx]

        df_train.to_csv(save_path + '/split/' + split + '/train_fold_%s_seed_%s.csv' % (fold, seed))
        df_val.to_csv(save_path + '/split/' + split + '/val_fold_%s_seed_%s.csv' % (fold, seed))

    return


if __name__ == '__main__':
    get_train_val_split()
    dataset = PlantDataset(df=pd.read_csv("../plant-pathology-2020-fgvc7/train.csv"))
    a = dataset[0]
    # cv2.imshow('a', a[0])
    # cv2.waitKey(0)
    print(a[1])