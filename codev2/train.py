from model import *
from utils import *
from dataset import PlantDataset
from config import Config


def training(model,
             epoches,
             criterion,
             optimizer,
             scheduler,
             accumulate,
             data_train,
             data_valid,
             i_fold):
    scheduler_each_iter = Config.scheduler_each_iter

    best_loss, best_score = 100, 0.0

    for epoch in range(epoches):
        # Train
        model.train()
        start_time = time.time()
        train_loss = 0.0
        for step, (images, labels) in enumerate(data_train):
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(-1))
            with amp.scale_loss(loss / accumulate, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (step + 1) % accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler_each_iter:
                    scheduler.step()

            train_loss += loss.item() / len(data_train)

        if not scheduler_each_iter:
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, (images, labels) in enumerate(data_valid):
            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)

            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                outputs = model(images)

                loss = criterion(outputs, labels.squeeze(-1))

                val_loss += loss.item()/len(data_valid)
                preds = torch.softmax(outputs, dim=1).data.cpu()

                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)

        val_score = roc_auc_score(val_labels, val_preds, average='macro')
        print('epoch:{}, lr:{:.5f}, train_loss:{:.4f}, val_loss:{:.4f}, val_score:{:.6f}, time:{:.2f}s'.
              format(epoch, scheduler.get_lr()[0], train_loss, val_loss, val_score, time.time()-start_time))

        with open(Config.model_save_path + Config.model_name
                       + '/Fold_{}_log'.format(i_fold) + '.txt', 'a') as log:
            log.write(
                'epoch:{}, lr:{:}, train_loss:{:.4f}, val_loss:{:.4f}, val_score:{:.6f}, time:{:.2f}s \n'.
                format(epoch, scheduler.get_lr()[0], train_loss, val_loss, val_score, time.time()-start_time))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Config.model_save_path + Config.model_name
                       + '/Fold_{}_loss'.format(i_fold) + '.pth')
            print('best loss model saved!')

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path+Config.model_name
                       + '/Fold_{}_score'.format(i_fold) + '.pth')
            print('best score model saved!')

    return val_score


def train_fold(i_fold):
    print('-----------------Fold:{}, model:{}, training start---------------------'.
          format(i_fold, Config.model_name))

    train = pd.read_csv(os.path.join(Config.spilt_save_path,
                                     'split/{}/train_fold_{}_seed_{}.csv'.
                                     format(Config.spilt_method, i_fold, Config.split_seed)))
    train.reset_index(drop=True, inplace=True)

    valid = pd.read_csv(os.path.join(Config.spilt_save_path,
                                     'split/{}/val_fold_{}_seed_{}.csv'.
                                     format(Config.spilt_method, i_fold, Config.split_seed)))
    valid.reset_index(drop=True, inplace=True)

    dataset_train = PlantDataset(df=train, transform=Config.transforms_train)
    dataset_valid = PlantDataset(df=valid, transform=Config.transforms_valid)

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=Config.batch_size,
                                  num_workers=Config.num_workers,
                                  shuffle=True)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=Config.batch_size,
                                  num_workers=Config.num_workers,
                                  shuffle=False)

    model = PlantModel(num_classes=4)
    model.cuda()

    criterion = DenseCrossEntropy()

    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6, 10, 15, 75], gamma=0.25)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    training(model=model,
             epoches=20,
             criterion=criterion,
             optimizer=optimizer,
             scheduler=scheduler,
             accumulate=1,
             data_train=dataloader_train,
             data_valid=dataloader_valid,
             i_fold=i_fold)


if __name__ == '__main__':
    os.makedirs(Config.model_save_path+Config.model_name, exist_ok=True)

    for i in range(5):
        train_fold(i)
