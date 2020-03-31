from library import *
from dataset import PlantDataset, get_train_val_split
from model import PlantModel
from config import Config

DEBUG = False


def training(i_fold):
    if DEBUG:
        print(
        os.path.join(Config.spilt_save_path,
                     'split/{}/train_fold_{}_seed_{}.csv'.
                     format(Config.spilt_method, i_fold, Config.split_seed)))

    train = pd.read_csv(os.path.join(Config.spilt_save_path,
                                     'split/{}/train_fold_{}_seed_{}.csv'.
                                     format(Config.spilt_method, i_fold, Config.split_seed)))
    train.reset_index(drop=True, inplace=True)

    valid = pd.read_csv(os.path.join(Config.spilt_save_path,
                                     'split/{}/val_fold_{}_seed_{}.csv'.
                                     format(Config.spilt_method, i_fold, Config.split_seed)))
    valid.reset_index(drop=True, inplace=True)

    dataset_train = PlantDataset(df=train, transforms=Config.transforms_train)
    dataset_valid = PlantDataset(df=valid, transforms=Config.transforms_valid)

    dataloader_train = DataLoader(dataset_train, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=False)

    model = PlantModel(num_classes=4)
    model.cuda()
    if DEBUG:
        print(next(model.parameters()).is_cuda)

    criterion = DenseCrossEntropy()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    lr = 0.00005
    WEIGHT_DECAY = 0.00001
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'lr': lr,
         'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'lr': lr,
         'weight_decay': 0.0}
    ]

    optimizer = optim.Adam(optimizer_grouped_parameters)

    train_fold_results = []
    best_cv = 0

    for epoch in range(Config.epoches):
        print(epoch+1)
        model.train()
        tr_loss = 0

        for step, batch in enumerate(dataloader_train):
            images = batch[0]
            labels = batch[1]

            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)


            loss = criterion(outputs, labels.squeeze(-1))
            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

            if (step+1) % 10 == 0:
                _labels = (labels.data.cpu()).squeeze(-1).data.numpy()
                _prediction = (outputs.data.cpu()).data.numpy()
                try:
                    score = cal_score(_labels, _prediction)
                    print('Training step:{}, loss:{:.3f}, metrics:{:.3f}, healthy:{:.3f}, multiple:{:.3f}, rust:{:.3f}, scab:{:.3f}'.
                          format(step+1, tr_loss/(step+1), score[0], score[1], score[2], score[3], score[4]))
                except ValueError:
                    pass

        # Validate
        model.eval()
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, batch in enumerate(dataloader_valid):

            images = batch[0]
            labels = batch[1]

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)

            images = images.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                outputs = model(images)

                loss = criterion(outputs, labels.squeeze(-1))
                val_loss += loss.item()

                if val_preds is None:
                    val_preds = outputs
                else:
                    val_preds = torch.cat((val_preds, outputs), dim=0)

        val_labels, val_preds = val_labels.data.numpy(), (val_preds.data.cpu()).data.numpy()
        print(val_preds)
        cv_score = cal_score(val_labels, val_preds)
        print('Validation epoch:{}, val_loss:{:.3f}, metrics:{:.3f}, healthy:{:.3f}, multiple:{:.3f}, rust:{:.3f}, scab:{:.3f}'.
              format(epoch + 1, val_loss / dataloader_valid.__len__(),
                     cv_score[0], cv_score[1], cv_score[2], cv_score[3], cv_score[4]))

        if cv_score[0] > best_cv:
            print('scores improve from {:.3f} to {:.3f}, save model!'.format(best_cv, cv_score[0]))
            if best_cv != 0:
                os.remove(os.path.join(Config.model_save_path,
                                        'model/{}_epoch{}_fold{}_cv{}.pth'.format(
                                        Config.model_name, Config.epoches, i_fold, best_cv)))

            torch.save(model.state_dict(),
                       os.path.join(Config.model_save_path,
                                    'model/{}_epoch{}_fold{}_cv{}.pth'.format(
                                    Config.model_name, Config.epoches, i_fold, cv_score[0])))
            best_cv = cv_score[0]
        else:
            print('scores does not improve, best cv score:{:.3f}'.format(best_cv))

    # Save oof
    oof_preds = pd.DataFrame(val_preds)
    oof_preds.to_csv(os.path.join(Config.oof_save_path, 'oof/oof_fold_{}_seed_{}.csv'.
                                  format(i_fold, Config.split_seed)))

    return val_preds, train_fold_results


if __name__ == '__main__':
    get_train_val_split(df_path=os.path.join(Config.data_path, "train.csv"),
                        save_path=Config.spilt_save_path,
                        n_splits=Config.n_split,
                        seed=Config.split_seed,
                        split=Config.spilt_method)
    seed_everything(Config.train_seed)
    val_preds, train_fold_results = training(0)


