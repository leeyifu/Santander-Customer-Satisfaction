import pandas as pd
import json 
from load_data import load_data
from model import into_model

#if __name__ == '__main__':

with open('config.json') as config_file:
	config = json.load( config_file )

train_df, test_df = load_data()
X = train_df.iloc[:,:-1].values
y = train_df['TARGET'].values
Xt = test_df.values

yt = into_model(X, y, Xt)

res = pd.DataFrame( {'ID':test_df.index.values, 'TARGET':yt})
res.to_csv('res.csv', index=False)




