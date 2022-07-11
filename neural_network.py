from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy


def read_data():
    CSV_file = pd.read_csv("inputs\glass.data")
    ID = LabelEncoder().fit_transform(CSV_file.values[:, -1])
    Norm_Data = pd.DataFrame(MinMaxScaler().fit_transform(CSV_file.values[:, 1:-1]))
    return ID, Norm_Data


new_ID, Data = read_data()
LOGISNUM = 6
DATA=train_test_split(Data,new_ID,test_size=0.2,stratify=new_ID)

def softmax(set):
    id_predect = []
    new_set = [numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0) for x in set]
    for item in new_set:
        highest, sol = -100, 0
        for iter, section in enumerate(item):
            (highest, sol) = (section, iter) if section > highest else (highest, sol)
        id_predect.append(sol)
    return id_predect


class NeuralNet:
    def __init__(self, Data=DATA, logis_num=LOGISNUM):
        self.training_ID = (Data[0], Data[2])
        self.test_ID = (Data[1], Data[3])
        self.function = softmax
        self.num_logis = logis_num
        self.network = MLPClassifier(max_iter=10000)

    def train_network(self):
        self.network.fit(self.training_ID[0], self.training_ID[1])
        predict_id = self.function(self.network.predict_proba(self.test_ID[0]))
        id_with_test = zip(self.test_ID[1], predict_id)
        micro, macro = self.micro_macro(id_with_test, 6)
        self.test_network()
        return micro / len(self.test_ID[0]), macro / self.num_logis

    def test_network(self):
        predict_test = self.network.predict(self.test_ID[0])
        conf = confusion_matrix(self.test_ID[1], predict_test)
        report = classification_report(self.test_ID[1], predict_test)
        return conf, report

    def show_stats(self):
        predict_train = self.network.predict(self.training_ID[1])
        conf = confusion_matrix(self.training_ID[1], predict_train)
        report = classification_report(self.training_ID[1], predict_train)
        return conf, report

    def micro_macro(self, set, number):
        logits, t_logits = [0] * number, [0] * number
        micro, macro = 0, 0

        for id, predict in set:
            t_logits[predict] = t_logits[predict] + 1 if predict == id else t_logits[predict]
            micro = micro + 1 if predict == id else micro
            logits[predict] += 1
        new_set = zip(logits, t_logits)
        macro = sum([t_log / log for log, t_log in new_set])
        return micro, macro
