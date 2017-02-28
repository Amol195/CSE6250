
# Do not use anything outside of the standard distribution of python
# when implementing this class
import math 

class LogisticRegressionSGD:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = eta
        self.weight = [0.0] * n_feature
        self.mu = mu

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        value = y*1.0 - 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
        for i in X:
            self.weight[i[0]] = self.weight[i[0]] * (1-2*self.mu*self.eta)
            if i[1]!=0.0:
                self.weight[i[0]] = self.weight[i[0]] + self.eta * value * i[1]
        pass

    def predict(self, X):
        """
        Predict 0 or 1 given X and the current weights in the model
        """
        return 1 if predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        """
        Sigmoid function
        """
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))