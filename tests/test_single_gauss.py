import os

from collections import defaultdict

from gmm_hmm_asr.data import DataTuple
from gmm_hmm_asr.trainers import SingleGaussTrainer

from .utils import *

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE=os.path.join(THIS_DIR, 'data/train_1digit.feat')
TEST_FILE=os.path.join(THIS_DIR, 'data/test_1digit.feat')

DIGITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "z", "o"]


def test_single_gaussian():
    train_data = get_data_dict(TRAIN_FILE)
    train_data = {key:train_data[key] for key in list(train_data.keys())[:100]}

    test_data = get_data_dict(TEST_FILE)
    test_data = {key:test_data[key] for key in list(test_data.keys())[:100]}

    data = []
    for key, feats in train_data.items():
        data.append(
            DataTuple(key=key, feats=feats, label=key.split('_')[1][0])
        )   
    ndim = data[0].feats.shape[1]

    sg_model = SingleGaussTrainer(ndim, DIGITS)
    sg_model.train(data)

    data = []
    for key, feats in test_data.items():
        data.append(
            DataTuple(key=key, feats=feats, label=key.split('_')[1][0])
        )
    preds = sg_model.predict(data)

    correct = 0
    for utt, pred in zip(data, preds):
        if pred[0] == utt.label:
            correct += 1

    accuracy = float(correct)/len(data) * 100
    print(f"Accuracy: {accuracy}")
    assert (accuracy >= 80 and accuracy <= 100)
