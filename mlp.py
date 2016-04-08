#coding:utf-8
# Theano v 0.7
# Keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
from data import load_data

if __name__=="__main__":
	print "=============load data============"
	data,label,test_data,test_id = load_data()
	#label = np_utils.to_categorical(label, 2)
	model = Sequential()
	model.add(Dense(32,input_dim=306,init='uniform'))
	model.add(Activation('relu'))
	model.add(Dense(64,init='uniform'))
	model.add(Activation('relu'))
	model.add(Dense(32,init='uniform'))
	model.add(Activation('relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	#print 'show Shape'
	###check the structure
	# x = 0
	# for i in model.layers:
	#     print 'the ',x,' layer'
	#     x = x+1
	#     print i.input_shape
	#     print i.output_shape
	#print 'Setting SGD'
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy',optimizer=sgd)
	#model.compile(loss='binary_crossentropy',optimizer='rmsprop')
	print 'Training'
	model.fit(data, label,show_accuracy=True,validation_split=0.1)
	#test_data = load_test()
	a = model.predict_proba(test_data)
	out = open('sub.csv','a')
	for i in range(len(test_id)):
		out.write(str(test_id[i])+','+str(a[i])+'\n')
	out.close()