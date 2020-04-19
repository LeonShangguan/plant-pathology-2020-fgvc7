from library import *
from config import Config
from model import PlantModel
from dataset import PlantDataset


def inference(model,
              data_test):
    model.eval()
    test_preds = None

    for step, (images, _) in enumerate(data_test):
        images = images.cuda()

        with torch.no_grad():
            outputs = model(images)

            if test_preds is None:
                test_preds = outputs.data.cpu()
            else:
                test_preds = torch.cat((test_preds, outputs.data.cpu()), dim=0)
    return test_preds


if __name__ == '__main__':
    submission_df = pd.read_csv(Config.data_path + 'sample_submission.csv')
    submission_df.iloc[:, 1:] = 0

    dataset_test = PlantDataset(df=submission_df, transform=Config.transforms_valid)
    data_test = DataLoader(dataset_test, batch_size=128, num_workers=16)

    model = PlantModel(num_classes=4)
    model.cuda()
    model.load_state_dict(torch.load('../output/se_resnext101Fold_0_score.pth'))
    test_preds = inference(model, data_test)
    print(submission_df.head())
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_preds, dim=1)
    submission_df.to_csv('submission_fold_{}.csv'.format(1), index=False)
