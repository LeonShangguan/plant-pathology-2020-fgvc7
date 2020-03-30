from library import *
from dataset import PlantDataset
from model import PlantModel
from config import Config


def load(model, params, skip=[]):
    pretrain_state_dict = torch.load(params)
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        if any(s in key for s in skip): continue
        try:
            state_dict[key] = pretrain_state_dict[key]
        except:
            print(key)
    model.load_state_dict(state_dict)

    return model


def inference():
    pass


if __name__ == "__main__":
    test = pd.read_csv('../plant-pathology-2020-fgvc7/sample_submission.csv')
    test.reset_index(drop=True, inplace=True)
    dataset_test = PlantDataset(df=test, transforms=Config.transforms_valid)
    dataloader_test = DataLoader(dataset_test, batch_size=64, num_workers=16)

    model = PlantModel(num_classes=4)
    model.cuda()
    model = load(model, 'checkpoint_filepath.pth')

    print(model)

    model.eval()

    test_preds = None

    for step, batch in enumerate(dataloader_test):
        print(step)
        images = batch[0]
        images = images.cuda()

        with torch.no_grad():
            outputs = model(images)

            if test_preds is None:
                test_preds = outputs.data.cpu()
            else:
                test_preds = torch.cat((test_preds, outputs.data.cpu()), dim=0)

    # Save predictions per fold
    test[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_preds, dim=1)
    test.to_csv('submission_fold_{}.csv'.format(0), index=False)
