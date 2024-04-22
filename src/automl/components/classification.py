from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, RandomizedSearchCV
import numpy as np


class ClassificationTrainer:
    
    def __init__(self, X, y):
        self.x = X
        self.y = y
    
    def train(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model_params = {
            'C': np.logspace(start=-5, stop=3, num=100),
            'penalty': ['l1', 'l2'],
        }
        model = LogisticRegression()
        validator = RandomizedSearchCV(estimator=model, param_distributions=model_params, n_iter=50, cv=kf, verbose=2, random_state=42, n_jobs=-1)
        validator.fit(self.x, self.y)
        best_model = validator.best_estimator_
        best_params = validator.best_params_
        return best_model
