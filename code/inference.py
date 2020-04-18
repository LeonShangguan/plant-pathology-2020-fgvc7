from library import *
from dataset import PlantDataset
from model import PlantModel
from config import Config


def load(model, params, skip=[]):
    pretrain_state_dict = torch.load(params)
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        if any(s in key for s in skip):
            continue
        try:
            state_dict[key] = pretrain_state_dict[key]
        except:
            print(key)
    model.load_state_dict(state_dict)

    return model


def inference(fold):
    test = pd.read_csv('../plant-pathology-2020-fgvc7/sample_submission.csv')
    test.reset_index(drop=True, inplace=True)
    dataset_test = PlantDataset(df=test, transforms=Config.transforms_valid)
    dataloader_test = DataLoader(dataset_test, batch_size=128, num_workers=16)

    model = PlantModel(num_classes=4)
    model.cuda()
    model = load(model, '../output/resnet34/acc_model_{}.pth'.format(fold))

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

    test_preds = torch.softmax(test_preds, dim=1)
    return test_preds


if __name__ == "__main__":
    test = pd.read_csv('../plant-pathology-2020-fgvc7/sample_submission.csv')
    test.reset_index(drop=True, inplace=True)
    results = []
    for i in range(5):
        test_preds = inference(i)
        results.append(test_preds)
    # Save predictions per fold
    result = (results[0]+results[1]+results[2]+results[3]+results[4])/5
    test[['healthy', 'multiple_diseases', 'rust', 'scab']] = result
    test.to_csv('{}.csv'.format(Config.model_name), index=False)
