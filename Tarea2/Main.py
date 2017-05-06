import sklearn.neural_network as sk
import numpy as np

class Main:
    def __init__(self):
        self.ex = np.zeros((4400, 49))
        self.clases = [0] * 4400
        f = open("training.txt", "r")
        for (i, line) in enumerate(f):
            self.ex[i] = line.split(",")
            self.clases[i] = self.ex[i, 48]
        f.close()

        self.test = np.zeros((1100, 49))
        f = open("tester.txt", "r")
        for (i, line) in enumerate(f):
            self.test[i] = line.split(",")
        f.close()

        print

        classifier = sk.MLPClassifier(hidden_layer_sizes=25, validation_fraction=0.2, solver="adam", max_iter=1000000)
        classifier.fit(self.ex[:,:47], self.clases)

        self.res = classifier.predict(self.test[:,:47])

        conf_mat_vals = np.zeros((11,11))
        for i in range(len(self.res)):
            conf_mat_vals[int(self.res[i]) - 1, int(self.test[i,48]) - 1] += 1

        print conf_mat_vals

        sum = 0
        for i in range(len(conf_mat_vals)):
            sum += conf_mat_vals[i, i]

        print sum

a = Main()