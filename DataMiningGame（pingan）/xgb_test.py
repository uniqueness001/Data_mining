import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# # Split the Learning Set
X_fit, X_eval, y_fit, y_eval= train_test_split(
    train, target, test_size=0.2, random_state=1
)

clf = xgb.XGBClassifier(missing=np.nan, max_depth=6,
                        n_estimators=5, learning_rate=0.15,
                        subsample=1, colsample_bytree=0.9, seed=1400)

# fitting
clf.fit(X_fit, y_fit, early_stopping_rounds=50, eval_metric="logloss", eval_set=[(X_eval, y_eval)])
#print y_pred
y_pred= clf.predict_proba(test)[:,1]