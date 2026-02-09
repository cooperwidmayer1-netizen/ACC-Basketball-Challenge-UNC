import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from .config import MAX_MARGIN_CAP, BLOWOUT_START, BLOWOUT_WEIGHT

def capped_mae(y_true, y_pred):
    err = np.abs(y_true - y_pred)
    err = np.minimum(err, MAX_MARGIN_CAP)
    return err.mean()

def fit_ridge(X, y, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model
