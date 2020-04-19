from library import *
from dataset import *
from criterion import *
from scheduler import *
from utils import *
from model import *
from optimizer import *

DEBUG = False


def train_model(model,
                optimizer,
                scheduler,
                data_loader,
                criterion,
                accumulate):
    model.train()
    avg_loss = 0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.cuda(), labels.cuda()

        output_train = model(images)

        loss = criterion(output_train, labels.squeeze(-1))

        with amp.scale_loss(loss/accumulate, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (step+1) % accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        avg_loss += loss.item() / len(data_loader)
    return avg_loss


def test_model(model,
               criterion,
               data_loader,
               mode='test'):
    val_preds = None
    val_labels = None
    avg_val_loss = 0.

    model.eval()

    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)

            outputs = model(images)

            avg_val_loss += (criterion(outputs, labels.squeeze(-1)).item() / len(data_loader))

            if val_preds is None:
                val_preds = outputs
            else:
                val_preds = torch.cat((val_preds, outputs), dim=0)

    val_labels, val_preds = (val_labels.data.cpu()).data.numpy(), (val_preds.data.cpu()).data.numpy()

    acc = cal_score(val_labels, val_preds)

    if mode == 'oof':
        return val_labels, val_preds
    else:
        return avg_val_loss, acc


def train_fold(
        fold,
        model,
        epochs,
        optimizer,
        scheduler,
        data_loader_train,
        data_loader_valid,
        criterion,
        accumulate):
    print('fold:{}, model:{}, accumulate:{}'.format(fold, Config.model_name, accumulate))
    best_avg_loss = 100.0
    best_acc = 0.0

    ### training
    for epoch in range(epochs):
        print('epoch:{}, lr:{}'.format(epoch, scheduler.get_lr()[0]))
        # scheduler.step()

        start_time = time.time()
        avg_loss = train_model(model,
                               optimizer,
                               scheduler,
                               data_loader_train,
                               criterion,
                               accumulate)
        avg_val_loss, acc = test_model(model,
                                       criterion,
                                       data_loader_valid)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_overall_acc={:.4f} \t'
               'val_healthy={:.4f}, val_multi={:.4f}, val_rust={:.4f}, val_scab={:.4f},'
               'time={:.2f}s'.format(epoch + 1, epochs, avg_loss, avg_val_loss, acc[0],
                                     acc[1], acc[2], acc[3], acc[4], elapsed_time))

        if avg_val_loss < best_avg_loss:
            best_avg_loss = avg_val_loss
            torch.save(model.state_dict(), '../output/{}/loss_model_{}.pth'.format(
                Config.model_name, str(fold)))
            print('Best Loss model saved!')

        if acc[0] > best_acc:
            best_acc = acc[0]
            torch.save(model.state_dict(), '../output/{}/acc_model_{}.pth'.format(
                Config.model_name, str(fold)))
            print('Best Acc model saved!')

        print('=================================')

    print('best loss:', best_avg_loss, 'best accuracy:', best_acc)

    return best_avg_loss, best_acc


def training(i_fold):
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

    criterion = DenseCrossEntropy()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.00005)

    ############         SCHEDULER        ##############
    T = len(dataloader_train) // 4 * 10  # cycle
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=T, T_mult=2, eta_max=1e-5, T_up=T // 10, gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6, 10, 15, 75], gamma=0.25)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    best = train_fold(i_fold,
                      model,
                      Config.epoches,
                      optimizer,
                      scheduler,
                      dataloader_train,
                      dataloader_valid,
                      criterion,
                      Config.accumulate)

    oof_label, oof_pred = test_model(model,
                                     criterion,
                                     dataloader_valid,
                                     'oof')
    print(oof_pred.shape)
    # Save oof
    df = pd.DataFrame({'label': list(oof_label), 'pred': list(oof_pred)})
    df.to_csv(os.path.join(Config.oof_save_path,
                           'oof/{}_fold_{}_seed_{}.csv'.format(
                               Config.model_name, i_fold, Config.split_seed)))

    return oof_pred


if __name__ == '__main__':
    os.makedirs('../output/{}'.format(Config.model_name), exist_ok=True)
    get_train_val_split(df_path=os.path.join(Config.data_path, "train.csv"),
                        save_path=Config.spilt_save_path,
                        n_splits=Config.n_split,
                        seed=Config.split_seed,
                        split=Config.spilt_method)
    seed_everything(Config.train_seed)
    for i in range(5):
        oof_preds = training(i)

