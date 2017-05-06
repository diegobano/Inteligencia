import numpy as np
import matplotlib.pyplot as plt


class Bayes:
    def __init__(self, ex, ex_1, ex_0, test):

        self.cov_mat_0 = np.cov(ex_0, rowvar=False)
        self.exp_vec_0 = [np.mean(ex_0[:, i]) for i in range(16)]

        self.cov_mat_1 = np.cov(ex_1, rowvar=False)
        self.exp_vec_1 = [np.mean(ex_1[:, i]) for i in range(16)]
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
        #print test[0, :]

        d_cr = []
        d_sr = []
        for i in range(test_data):
            p_cr = self.Gauss_0(test[i, 0:16])
            p_sr = self.Gauss_1(test[i, 0:16])
            d_cr.append(p_cr)
            d_sr.append(p_sr)
            # print "razon", razon, "theta", theta[roc]
        #print "d_cr", d_cr
        #print "d_sr", d_sr
        for roc in range(len(theta)):
            #print roc
            for i in range(test_data):
                p_sr = d_sr[i]
                p_cr = d_cr[i]
                if p_sr <= 0 and p_cr > 0:
                    razon = float("inf")
                elif p_sr > 0 and p_cr <= 0:
                    razon = 0
                elif p_sr == 0 and p_cr == 0:
                    continue
                else:
                    razon = p_cr / p_sr
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

    def Gauss_0(self, char):
        det = np.linalg.det(self.cov_mat_0)
        inv = np.linalg.inv(self.cov_mat_0)
        n = len(self.exp_vec_0)
        x_u_t = np.add(char, np.multiply(self.exp_vec_0, -1))
        x_u = np.transpose(x_u_t)

        return 1.0 / np.sqrt(2 * np.power(np.pi, n) * det) * \
               np.exp(-1.0 / 2 *
                      np.dot(np.dot(x_u_t, inv), x_u))

    def Gauss_1(self, char):
        det = np.linalg.det(self.cov_mat_1)
        inv = np.linalg.inv(self.cov_mat_1)
        n = len(self.exp_vec_1)
        x_u_t = np.add(char, np.multiply(self.exp_vec_1, -1))
        x_u = np.transpose(x_u_t)
        #print "inv", np.dot(np.dot(x_u_t, inv), x_u)

        res = 1.0 / np.sqrt(2 * np.power(np.pi, n) * det) * np.exp(
            -1.0 / 2 * np.dot(np.dot(x_u_t, inv), x_u))
        #print res
        return res

    def plot(self):
        plt.plot(self.t_vp[0, :], self.t_fp[0, :])
        plt.ylabel("Tasa de deteccion")
        plt.xlabel("Tasa de falsa deteccion")
        plt.title("Clasificador Bayesiano multidimensional")
        plt.show()

        #print "t_vp", self.t_vp,"t_fn", 1-self.t_vp
        #print "t_fp", self.t_fp, "t_vn", 1-self.t_fp