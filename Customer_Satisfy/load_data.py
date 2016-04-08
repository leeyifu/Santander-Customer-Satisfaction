import pandas as pd

def load_data(num = None):
	train_path = '../data/train.csv'
	test_path = '../data/test.csv'
	
	train_df = pd.read_csv(train_path, nrows = num, index_col=0)
	test_df = pd.read_csv(test_path, nrows = num, index_col=0)
	
	return train_df, test_df
	