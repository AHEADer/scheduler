from joblib import load
from sklearn import linear_model
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ


class Estimator:
    def __init__(self):
        self.resnet_alpha = load('.\\model\\alpha.joblib')
        self.resnet_beta = load('.\\model\\beta.joblib')

    def resnet_predict(self, train_f, epsilon):
        alpha = self.resnet_alpha.predict(train_f)
        beta = self.resnet_alpha.predict(train_f)
        for i in range(10, 1000):
            if stats.gamma.cdf(i, a=alpha, scale=beta) > 1-epsilon:
                # TODO change (10-5) to a parameter for scheduler
                return i+5

        return -1
