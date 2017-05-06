import sklearn.preprocessing as skp
import sklearn.svm as sks
import numpy as np


class Main:
    def __init__(self):
        self.ex = np.zeros((4400, 49))
        self.clases = [0] * len(self.ex)
        f = open("training.txt", "r")
        for (i, line) in enumerate(f):
            self.ex[i] = line.split(",")
            self.clases[i] = 0 if self.ex[i, 48] <= 1 else 1
        f.close()

        self.test = np.zeros((1100, 49))
        f = open("tester.txt", "r")
        for (i, line) in enumerate(f):
            self.test[i] = line.split(",")
        f.close()

        normalized_ex = skp.StandardScaler().fit_transform(self.ex[:, :47])
        print normalized_ex
        clf = sks.SVC(kernel='linear')
        clf.fit(normalized_ex, self.clases)

        score = clf.decision_function(self.test[:, :47])
        print score


a = Main()
