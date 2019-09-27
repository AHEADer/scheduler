from joblib import load
import scipy.stats as stats


class Estimator:
    def __init__(self):
        self.resnet_alpha = load('./model/alpha.joblib')
        self.resnet_theta = load('./model/theta.joblib')

    def resnet_predict(self, train_f, epsilon):
        alpha = self.resnet_alpha.predict([train_f])
        beta = self.resnet_theta.predict([train_f])
        for i in range(1, 1000):
            if stats.gamma.cdf(i, a=alpha, scale=beta) > 1-epsilon:
                # TODO change (10-5) to a parameter for scheduler
                return i+5

        return -1