import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

def into_model(X, y, Xt):
	res = xgbMachine(X, y, Xt)
	return res

def rfr( train_df):
	pass

def xgbMachine(X, y, Xt):
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
	clf = xgb.XGBClassifier(max_depth=5, n_estimators=400, learning_rate=0.05)
	clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=[(X_test, y_test)])
	print('XG2 test AUC:', roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
	yt = clf.predict_proba(Xt)[:,1]
	return yt
