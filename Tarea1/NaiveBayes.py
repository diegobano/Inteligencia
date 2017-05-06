import numpy as np
import math
import matplotlib.pyplot as plt
import random

class NaiveBayes:
    def __init__(self, ex, ex_1, ex_0, test):
        test_1 = []
        test_0 = []
        for i in test:
            if i[16] == 1:
                test_1.append(i)
            else:
                test_0.append(i)

        bins = [10]*16

        hists_1 = np.zeros((16, max(bins)))
        ranges_1 = np.zeros((16, max(bins) + 1))
        hists_0 = np.zeros((16, max(bins)))
        ranges_0 = np.zeros((16, max(bins) + 1))
        for i in range(16):
            (hists_1[i,0:bins[i]], ranges_1[i,0:bins[i]+1]) = np.histogram(ex_1[:, i], bins[i])
            (hists_0[i,0:bins[i]], ranges_0[i,0:bins[i]+1]) = np.histogram(ex_0[:, i], bins[i])

        for i in range(16):
            hists_0[i] = np.multiply(hists_0[i], 1.0 / len(ex_0))
            hists_1[i] = np.multiply(hists_1[i], 1.0 / len(ex_1))

        test_data = len(test)

        theta = np.linspace(.1, 100000, 2000)

        vp = np.zeros((1, len(theta)))
        self.t_vp = np.zeros((1, len(theta)))
        fp = np.zeros((1, len(theta)))
        self.t_fp = np.zeros((1, len(theta)))
        vn = np.zeros((1, len(theta)))
        self.t_vn = np.zeros((1, len(theta)))
        fn = np.zeros((1, len(theta)))
        self.t_fn = np.zeros((1, len(theta)))

        #print theta

        d_cr = []
        d_sr = []
        for i in range(test_data):
            cr = [0,0]
            sr = [0,0]
            p_cr = 1
            for j in range(16):
                if test[i, j] < ranges_1[j, 0] or test[i, j] > ranges_1[j,bins[j]-1]:
                    p_cr = 0
                    break
                if test[i, j] == ranges_1[j,bins[j]-1]:
                    p_cr *= hists_1[j, bins[j]-1]
                    continue
                for k in range(bins[j]):
                    if ranges_1[j, k] <= test[i, j] < ranges_1[j, k + 1]:
                        cr = [j,k]
                        p_cr *= hists_1[j, k]
            d_cr.append(p_cr)

            p_sr = 1
            for j in range(16):
                if test[i, j] < ranges_0[j, 0] or test[i, j] > ranges_0[j,bins[j]-1]:
                    p_sr = 0
                    break
                if test[i, j] == ranges_0[j,bins[j]-1]:
                    p_sr *= hists_0[j, bins[j]-1]
                    continue
                for k in range(bins[j]):
                    if ranges_0[j, k] <= test[i, j] < ranges_0[j, k + 1]:
                        sr = [j,k]
                        p_sr *= hists_0[j, k]
            d_sr.append(p_sr)
            #if p_sr <= 0 and p_cr <= 0:
                #print cr, sr

        for roc in range(len(theta)):
            #print roc
            for i in range(len(d_sr)):
                p_sr = d_sr[i]
                p_cr = d_cr[i]
                if p_sr <= 0 and p_cr > 0:
                    razon = float("inf")
                elif p_sr > 0 and p_cr <= 0:
                    razon = 0
                elif p_sr <= 0 and p_cr <= 0:
                    #print cr, sr
                    continue
                else:
                    razon = p_cr/p_sr
                #print "razon", razon, "theta", theta[roc]
                choice = 1 if (razon >= theta[roc]) else 0
                if choice == 1:
                    if test[i, 16] == 1:
                        vp[0, roc] += 1
                    else:
                        fp[0, roc] += 1
                else:
                    if test[i, 16] == 1:
                        fn[0, roc] += 1
                    else:
                        vn[0, roc] += 1

            self.t_vp[0, roc] = vp[0, roc] / (vp[0,roc] + fn[0,roc])
            self.t_fp[0, roc] = fp[0, roc] / (fp[0,roc] + vn[0,roc])


    def plot(self):
        plt.plot(self.t_vp[0, :], self.t_fp[0, :])
        plt.title("Clasificador Bayesiano utilizando Naive Bayes")
        plt.ylabel("Tasa de deteccion")
        plt.xlabel("Tasa de falsa deteccion")
        plt.show()

        #print "t_vp", self.t_vp,"t_fn", 1-self.t_vp
        #print "t_fp", self.t_fp, "t_vn", 1-self.t_fp