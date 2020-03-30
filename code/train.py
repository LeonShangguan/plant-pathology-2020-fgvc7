from library import *
from dataset import PlantDataset
from model import PlantModel
from config import Config


def training(i_fold, model, criterion, optimizer, dataloader_train, dataloader_valid):
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
            print(step)

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

                preds = torch.softmax(outputs, dim=1).data.cpu()

                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)
        cv_score = roc_auc_score(val_labels, val_preds)
        print("CV score: {:.4f}".format(cv_score))
        if cv_score > best_cv:
            torch.save(model.state_dict(),
                       '../output/model/{}_epoch{}_fold{}_cv{}.pth'.format(
                           Config.model_name, Config.epoches, i_fold, cv_score))

        train_fold_results.append({
            'fold': i_fold,
            'epoch': epoch,
            'train_loss': tr_loss / len(dataloader_train),
            'valid_loss': val_loss / len(dataloader_valid),
            'valid_score': roc_auc_score(val_labels, val_preds, average='macro'),
        })
    return val_preds, train_fold_results


if __name__ == '__main__':
    train = pd.read_csv('../output/split/StratifiedKFold/train_fold_0_seed_960630.csv')
    train.reset_index(drop=True, inplace=True)

    valid = pd.read_csv('../output/split/StratifiedKFold/val_fold_0_seed_960630.csv')
    valid.reset_index(drop=True, inplace=True)

    dataset_train = PlantDataset(df=train, transforms=Config.transforms_train)
    dataset_valid = PlantDataset(df=valid, transforms=Config.transforms_valid)

    dataloader_train = DataLoader(dataset_train, batch_size=16, num_workers=20, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=16, num_workers=20, shuffle=False)

    model = PlantModel(num_classes=4)
    model.cuda()

    print(next(model.parameters()).is_cuda)

    criterion = DenseCrossEntropy()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    lr = 0.00005
    WEIGHT_DECAY = 0.00001
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], \
         'lr': lr,
         'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], \
         'lr': lr,
         'weight_decay': 0.0}
    ]

    optimizer = optim.Adam(optimizer_grouped_parameters)
    val_preds, train_fold_results = training(0, model, criterion, optimizer, dataloader_train, dataloader_valid)

    oof_preds = np.zeros((valid.shape[0], 4))
    oof_preds = val_preds.numpy()
    print(oof_preds)

    print("5-Folds CV score: {:.4f}".format(roc_auc_score(valid.iloc[:, 2:6].values, oof_preds, average='macro')))
