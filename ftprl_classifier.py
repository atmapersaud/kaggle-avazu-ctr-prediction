import numpy as np

class Ftprl_classifier(object):
    def __init__(self, alpha, beta, lambda_2, lambda_1, num_params):
        self.num_params = num_params
        self.inter1 = np.zeros(num_params)
        self.inter2 = np.zeros(num_params)
        self.wvec = {}
        self.alpha = alpha
        self.beta = beta
        self.lambda_2 = lambda_2
        self.lambda_1 = lambda_1

    def weight_update(x, y, yhat):
        grad = yhat - y
        for d in dimensions(x):
            sig = (sqrt(n[d] + pow(grad,2)) - sqrt(inter1[d])) / alpha
            inter1[d] = grad - (sig * wvec[d]) + inter1[d]
            inter2[d] = pow(grad,2) + inter1[d]

    def predict_ctr(self, x):
        wvec = {}
        dotprod = 0
        for d in dimensions(x):
            if inter2[d] > 0:
                sgn = 1
            else:
                sgn = -1
            if inter2[d]*sgn <= lambda_1:
                wvec[d] = 0
            else:
                wvec[d] = (sgn*lambda_1 - inter2[d]) / ((beta + sqrt(inter1[d])) / alpha + lambda_2)
            dotprod += wvec[d]

        self.wvec = wvec
        return bounded_logsig(dotprod)
