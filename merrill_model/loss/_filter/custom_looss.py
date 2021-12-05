#
import numpy


#


#

def logcoshlog1p(y_true, y_pred):
    grad = -(1 / (1 + y_pred)) * (numpy.tanh(numpy.log((1 + y_true) / (1 + y_pred))))
    hess = (1 / numpy.power((1 + y_pred), 2)) * (numpy.tanh(numpy.log((1 + y_true) / (1 + y_pred))) + 1 / (numpy.power(numpy.cosh(numpy.log((1 + y_true) / (1 + y_pred))), 2)))
    return grad, hess
