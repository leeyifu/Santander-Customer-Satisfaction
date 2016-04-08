#coding:utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')
	# remove constant columns
	remove = []
	for col in train.columns:
		###std() refers to standard deviation
		if train[col].std() == 0:
			remove.append(col)
	###axis = 1 for col and 0 for row, inplace means replace the original data
	train.drop(remove,axis=1,inplace=True)
	test.drop(remove,axis=1,inplace=True)

	# remove duplicated columns
	remove = []
	c = train.columns
	for i in range(len(c)-1):
		v = train[c[i]].values
		for j in range(i+1,len(c)):
			if np.array_equal(v, train[c[j]].values):
				remove.append(c[j])
	train.drop(remove,axis=1,inplace=True)
	test.drop(remove,axis=1,inplace=True)

	train_label = train['TARGET'].values
	train_data = train.drop(['id','TARGET'],axis=1).values

	test_id = test['id']
	test_data = test.drop(['id'],axis=1).values

	# Scale data
	scaler = StandardScaler()
	scaler.fit(train_data)
	train_data = scaler.transform(train_data)

	# apply same transformation to test data
	test_data = scaler.transform(test_data)
	return train_data,train_label,test_data,test_id








# def load_train():
# 	f = open('train.csv','r')
# 	data = np.empty((76020,369),dtype="float32")
# 	label = np.empty((76020),dtype="uint8")
# 	#read 1st line
# 	f.readline()
# 	#########76030 samples with 369 features##########
# 	i = 0
# 	for line in f:
# 		temp = line.split(',')
# 		data[i,:] = np.asarray(temp[1:-1],dtype="float32")
# 		label[i] = temp[-1]
# 		i = i+1
# 	f.close()
# 	return data,label

# def load_test():
# 	f = open('test.csv','r')
# 	data = np.empty((75818,369),dtype="float32")
# 	#read 1st line
# 	f.readline()
# 	i = 0
# 	for line in f:
# 		temp = line.split(',')
# 		data[i,:] = np.asarray(temp[1:],dtype="float32")
# 		i = i+1
# 	f.close()
# 	return data
# def build_data():
# 	data = np.empty((7000,370),dtype="float32")
# 	label = np.empty((7000),dtype="uint8")
# 	zero = np.empty((73012,369),dtype="float32")
# 	one = np.empty((3008,369),dtype="float32")
# 	f = open('train.csv','r')
# 	f.readline()
# 	i = 0
# 	z_i = 0
# 	o_i = 0
# 	for line in f:
# 		temp = line.split(',')
# 		#data[i,:] = 
# 		#label[i] = temp[-1]
# 		if temp[-1]==0:
# 			zero[z_i,:] = np.asarray(temp[1:-1],dtype="float32")
# 			z_i = z_i+1
# 		elif temp[-1]==1:
# 			one[o_i,:] = np.asarray(temp[1:-1],dtype="float32")
# 			o_i = o_i + 1
# 		i = i+1
# 	f.close()
# 	data[0:3008,:-1] = one
# 	data[0:3008,-1] = 1
# 	data[3008:,:-1] = zero[0:3992]
# 	data[3008:,:-1] = 0
# 	np.random.shuffle(data)
# 	label = data[:,-1]
# 	data = data[:,:-1]
# 	return data,label
