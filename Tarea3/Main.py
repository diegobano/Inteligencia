import sklearn.preprocessing as skp
import sklearn.svm as sks
import numpy as np
import matplotlib.pyplot as plt



class Main:
    def __init__(self):
        self.ex = np.zeros((4400, 49))
        self.classes = np.zeros(4400)
        f = open("training.txt", "r")
        for (i, line) in enumerate(f):
            self.ex[i] = line.split(",")
            self.classes[i] = 0 if self.ex[i, 48] <= 1 else 1
        f.close()

        self.test = np.zeros((1100, 49))
        f = open("tester.txt", "r")
        for (i, line) in enumerate(f):
            self.test[i] = line.split(",")
        f.close()
        self.ss = skp.StandardScaler()
        self.ss.fit(self.ex[:, :47])

        self.norm_test = self.ss.transform(self.test[:, :47])

        lin_scores = self.classify('linear')
        pol_scores = self.classify('poly')
        rbf_scores = self.classify('rbf')

        thresh_num = 1000

        lin_thresh = np.linspace(min(lin_scores), max(lin_scores), thresh_num)
        lin_predict = np.zeros((thresh_num, len(self.test)))
        #print lin_scores, lin_thresh

        pol_thresh = []
        rbf_thresh = []

        for i in range(11):
            pol_thresh.append(np.linspace(min(pol_scores[i]), max(pol_scores[i]), thresh_num))
            rbf_thresh.append(np.linspace(min(rbf_scores[i]), max(rbf_scores[i]), thresh_num))

        pol_predict = np.zeros((11, thresh_num, len(self.test)))
        rbf_predict = np.zeros((11, thresh_num, len(self.test)))

        for i in range(thresh_num):
            for j in range(len(lin_scores)):
                lin_predict[i, j] = 1 if lin_scores[j] > lin_thresh[i] else 0
                for k in range(11):
                    pol_predict[k, i, j] = 1 if pol_scores[k][j] > pol_thresh[k][i] else 0
                    rbf_predict[k, i, j] = 1 if rbf_scores[k][j] > rbf_thresh[k][i] else 0

        self.plot_roc(lin_predict, " linear SVC")
        for i in range(11):
            self.plot_roc(pol_predict[i], " poly SVC: " + str(i))
            self.plot_roc(rbf_predict[i], " rbf SVC: " + str(1.0 / (i + 42)))

    def plot_roc(self, predictions, type):
        conf_mats = np.zeros((len(predictions), 2, 2))
        for (i, sample) in enumerate(predictions):
            #print i, sample
            for j in range(len(sample)):
                conf_mats[i, (1 if int(self.test[j, 48]) > 1 else 0), int(sample[j])] += 1
        #print conf_mats
        roc_data = np.zeros((len(predictions), 2))

        for (i, mat) in enumerate(conf_mats):
            roc_data[i, 0] = mat[1, 1] / (mat[1, 0] + mat[1, 1])
            roc_data[i, 1] = mat[0, 1] / (mat[0, 1] + mat[0, 0])

        #print roc_data
        plt.xlabel("False positives")
        plt.ylabel("True positives")
        plt.title("ROC Curve" + type)
        plt.plot(roc_data[:, 1], roc_data[:, 0])
        plt.show()

    def classify(self, kernel):
        normalized_ex = self.ss.transform(self.ex[:, :47])
        if kernel == 'linear':
            clf = sks.SVC(kernel=kernel)
            clf.fit(normalized_ex, self.classes)
            score = clf.decision_function(self.norm_test)
        elif kernel == 'poly':
            degrees = 11
            clf = []
            score = []
            for i in range(degrees):
                clf.append(sks.SVC(kernel=kernel, degree=i+1))
                clf[i].fit(normalized_ex, self.classes)
                score.append(clf[i].decision_function(self.norm_test))

        elif kernel == 'rbf':
            gammas = np.linspace(len(self.test[0, :47]) - 20, len(self.test[0, :47]) + 20, 11)
            clf = []
            score = []
            for i in range(len(gammas)):
                clf.append(sks.SVC(kernel=kernel, gamma=1.0 / gammas[i]))
                clf[i].fit(normalized_ex, self.classes)
                score.append(clf[i].decision_function(self.norm_test))

        return score

a = Main()
