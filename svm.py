from data import load_data
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.svm import OneClassSVM
train_data,train_label,test_data,test_id = load_data()
pca = PCA(n_components=100)
pca.fit(train_data)
clf = OneClassSVM()
scores = cross_validation.cross_val_score(clf,train_data,train_label, scoring='roc_auc', cv=5) 
print(scores.mean())
#pca.transform(test_data)
