import numpy as np
import NaiveBayes
import Bayes

class Main:
    def __init__(self):
        self.ex = np.zeros((920, 17))
        self.l_1 = 0
        self.l_0 = 0
        f = open("examples.txt", "r")
        for (i, line) in enumerate(f):
            self.ex[i] = line.split(",")
            if self.ex[i, 16] == 1:
                self.l_1 += 1
            else:
                self.l_0 += 1
        f.close()

        self.test = np.zeros((231, 17))
        f = open("testing.txt", "r")
        for (i, line) in enumerate(f):
            self.test[i] = line.split(",")
        f.close()

        (self.ex_0, self.ex_1) = self.split(self.ex)

    def split(self, data):
        ex_1 = []
        ex_0 = []
        for i in data:
            if i[16] == 1:
                ex_1.append(i)
            else:
                ex_0.append(i)

        l_1 = len(ex_1)
        l_0 = len(ex_0)

        np_ex_1 = np.zeros((l_1, 16))
        for i in range(l_1):
            np_ex_1[i, :] = ex_1[i][0:16]

        ex_1 = np_ex_1

        np_ex_0 = np.zeros((l_0, 16))
        for i in range(l_0):
            np_ex_0[i, :] = ex_0[i][0:16]

        ex_0 = np_ex_0
        return (ex_0, ex_1)


def tarea1(entrenamiento, prueba):
    d = Main()
    (t_0, t_1) = d.split(entrenamiento)
    nb = NaiveBayes.NaiveBayes(entrenamiento, t_1, t_0, prueba)
    nb.plot()
    b = Bayes.Bayes(entrenamiento, t_1, t_0, prueba)
    b.plot()
    return

data = Main()
tarea1(data.ex, data.test)