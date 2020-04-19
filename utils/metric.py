import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def cal_score_train(labels, predictions):

    labels = labels.astype(np.int)
    predictions = predictions.astype(np.int)

    labels = np.argmax(labels, axis=1)
    predictions = np.argmax(predictions, axis=1)

    return accuracy_score(labels, predictions)


def cal_score(labels, predictions, predictions_softmax):

    labels = labels.astype(np.int)
    predictions = predictions.astype(np.int)

    healthy_true = labels[:, 0]
    multiple_diseases_true = labels[:, 1]
    rust_true = labels[:, 2]
    scab_true = labels[:, 3]

    healthy_pre_softmax = predictions_softmax[:, 0]
    multiple_diseases_pre_softmax = predictions_softmax[:, 1]
    rust_pre_softmax = predictions_softmax[:, 2]
    scab_pre_softmax = predictions_softmax[:, 3]

    healthy_auc = roc_auc_score(healthy_true, healthy_pre_softmax)
    multiple_diseases_auc = roc_auc_score(multiple_diseases_true, multiple_diseases_pre_softmax)
    rust_auc = roc_auc_score(rust_true, rust_pre_softmax)
    scab_auc = roc_auc_score(scab_true, scab_pre_softmax)

    labels = np.argmax(labels, axis=1)
    predictions = np.argmax(predictions, axis=1)

    avg_auc = (healthy_auc + multiple_diseases_auc + rust_auc + scab_auc) / 4.0
    avg_acc = accuracy_score(labels, predictions)

    return np.asarray([avg_auc, healthy_auc, multiple_diseases_auc, rust_auc, scab_auc, avg_acc])