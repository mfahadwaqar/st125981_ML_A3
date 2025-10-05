import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import mlflow


class LinearRegression(object):
    
    kfold = KFold(n_splits=3)
            
    def __init__(self, regularization=None, lr=0.001, method='batch', num_epochs=500,
                 batch_size=50, cv=kfold, momentum=False, beta=0.9, init_method="zeros"):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization

        # New parameters
        self.init_method = init_method
        self.momentum    = momentum
        self.beta        = beta
        self.velocity    = None

    # Evaluation metrics
    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    def r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_total)

    def _initialize_weights(self, n_features, method="zeros"):
        if method in ["zeros", "zero"]:
            return np.zeros(n_features)
        elif method == "xavier":
            limit = 1 / np.sqrt(n_features)
            return np.random.uniform(-limit, limit, size=n_features)
        else:
            raise ValueError(f"Initialization method '{method}' unknown")

    def fit(self, X_train, y_train):
        self.kfold_scores = []

        # Always convert y to numpy for consistency
        y_train = y_train.values if hasattr(y_train, "values") else y_train

        if self.cv is not None:  # --- Case 1: K-Fold CV ---
            for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
                # --- Split X ---
                if hasattr(X_train, "iloc"):   # Pandas
                    X_cross_train = X_train.iloc[train_idx].values
                    y_cross_train = y_train[train_idx]
                    X_cross_val   = X_train.iloc[val_idx].values
                    y_cross_val   = y_train[val_idx]
                else:                          # NumPy
                    X_cross_train = X_train[train_idx]
                    y_cross_train = y_train[train_idx]
                    X_cross_val   = X_train[val_idx]
                    y_cross_val   = y_train[val_idx]

                # Initialize weights (+1 for bias term)
                self.theta = self._initialize_weights(X_cross_train.shape[1] + 1, self.init_method)
                if self.momentum:
                    self.velocity = np.zeros_like(self.theta)

                # Train this fold
                self._train_on_data(X_cross_train, y_cross_train, val_data=(X_cross_val, y_cross_val), fold=fold)

        else:  
            X_train = X_train.values if hasattr(X_train, "values") else X_train
            self.theta = self._initialize_weights(X_train.shape[1] + 1, self.init_method)
            if self.momentum:
                self.velocity = np.zeros_like(self.theta)

            self._train_on_data(X_train, y_train)  # no validation here


    def _train_on_data(self, X, y, val_data=None, fold=None):
        # Ensure numpy arrays
        y = y.values if hasattr(y, "values") else y
        X = X.values if hasattr(X, "values") else X

        for epoch in range(self.num_epochs):
            # Shuffle
            perm = np.random.permutation(X.shape[0])
            X_shuff, y_shuff = X[perm], y[perm]

            if self.method == 'sto':
                for i in range(X_shuff.shape[0]):
                    X_batch = X_shuff[i].reshape(1, -1)
                    y_batch = np.array([y_shuff[i]])
                    self._train(X_batch, y_batch)

            elif self.method == 'mini':
                for i in range(0, X_shuff.shape[0], self.batch_size):
                    X_batch = X_shuff[i:i+self.batch_size]
                    y_batch = y_shuff[i:i+self.batch_size]
                    self._train(X_batch, y_batch)

            else:  # batch
                self._train(X_shuff, y_shuff)

            # --- Logging every epoch ---
            y_pred = self.predict(X)
            mse_epoch = mean_squared_error(y, y_pred)
            r2_epoch = r2_score(y, y_pred)

            # log training metrics
            mlflow.log_metric("train_mse", mse_epoch, step=epoch)
            mlflow.log_metric("train_r2", r2_epoch, step=epoch)

            # print to console every 10 epochs
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                if fold is not None:
                    print(f"[Fold {fold}] Epoch {epoch}/{self.num_epochs} - Train MSE: {mse_epoch:.6f}, R²: {r2_epoch:.6f}")
                else:
                    print(f"Epoch {epoch}/{self.num_epochs} - Train MSE: {mse_epoch:.6f}, R²: {r2_epoch:.6f}")

            # if validation is available (CV)
            if val_data is not None:
                X_val, y_val = val_data
                X_val = X_val.values if hasattr(X_val, "values") else X_val
                y_val = y_val.values if hasattr(y_val, "values") else y_val

                y_val_pred = self.predict(X_val)
                mse_val = mean_squared_error(y_val, y_val_pred)
                r2_val = r2_score(y_val, y_val_pred)

                mlflow.log_metric(f"val_mse_fold{fold}", mse_val, step=epoch)
                mlflow.log_metric(f"val_r2_fold{fold}", r2_val, step=epoch)

                if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                    print(f"[Fold {fold}] Epoch {epoch}/{self.num_epochs} - Val MSE: {mse_val:.6f}, R²: {r2_val:.6f}")

    def _train(self, X, y):
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]

        # Predictions
        yhat = X @ self.theta
        m = X.shape[0]

        # Gradient calculation
        grad = (1/m) * (X.T @ (yhat - y))

        # Add regularization if provided
        if self.regularization is not None:
            grad += self.regularization.derivation(self.theta)

        # ---- Gradient Clipping ----
        max_norm = 5.0
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_norm:
            grad = grad * (max_norm / grad_norm)

        # Parameter update (momentum or vanilla)
        if self.momentum:
            self.velocity = self.beta * self.velocity + (1 - self.beta) * grad
            self.theta = self.theta - self.lr * self.velocity
        else:
            self.theta = self.theta - self.lr * grad

        # Compute and return loss
        return self.mse(y, yhat)

    def predict(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = np.c_[np.ones(X.shape[0]), X]  # Adding bias
        return X @ self.theta

    def _coef(self):
        return self.theta[1:]  

    def _bias(self):
        return self.theta[0]

from sklearn.preprocessing import PolynomialFeatures

# Regularization classes
class NoRegularization:
    def derivation(self, theta):
        return np.zeros_like(theta)

class RidgeRegularization:
    def __init__(self, lamda=1e-3):
        self.lamda = lamda
    def derivation(self, theta):
        d = self.lamda * theta.copy()
        d[0] = 0.0
        return d

class LassoRegularization:
    def __init__(self, lam=1e-3):
        self.lam = lam
    def derivation(self, theta):
        d = self.lam * np.sign(theta)
        d[0] = 0.0
        return d

# Polynomial feature transformer
poly = PolynomialFeatures(degree=2, include_bias=False)

regularizations = {
    "normal": NoRegularization(),
    "ridge": RidgeRegularization(1e-3),
    "lasso": LassoRegularization(1e-3),
    "poly": poly
}