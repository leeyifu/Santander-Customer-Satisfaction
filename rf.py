#coding:utf-8
from data import load_data
from sklearn.ensemble import RandomForestClassifier 
from sklearn import cross_validation

train_data,train_label,test_data,test_id = load_data()
clf = RandomForestClassifier(n_estimators=100,max_depth=17,random_state=1)
scores = cross_validation.cross_val_score(clf,train_data,train_label, scoring='roc_auc', cv=5) 
print(scores.mean())

clf.fit(train_data,train_label)
predict = clf.predict_proba(test_data)
out = open('rf_3_23.csv','a')
for i in range(len(test_id)):
	out.write(str(test_id[i])+','+str(predict[i][1])+'\n')
out.close()