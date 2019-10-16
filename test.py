import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import csv

########### LOAD AND PREPROCESS ############
Train_1= pd.read_csv('C:/Users/Aakash Kamble/Desktop/Kaggle/tcdml1920-income-ind/tcd ml 2019-20 income prediction training (with labels).csv')
Train_1= Train_1.fillna(method='ffill')

Train_2= pd.read_csv('C:/Users/Aakash Kamble/Desktop/Kaggle/tcdml1920-income-ind/tcd ml 2019-20 income prediction test (without labels).csv')
Train_2=Train_2.fillna(method='ffill')

validation= Train_1.iloc[89594:,:]
training= Train_1.iloc[:89594,:]
predict= Train_2
#22399
#111993
#pad 73230

df_row_reindex = pd.concat([training, validation,predict])
print(df_row_reindex.head())

X = df_row_reindex[['Country','Age','Year of Record','Body Height [cm]','Profession']]
Y = df_row_reindex[['Income in EUR']]


le=LabelEncoder()
X['Country']=le.fit_transform(X['Country'])
X['Profession']=le.fit_transform(X['Profession'])

Hot_Encode=OneHotEncoder(categorical_features=[0])
X = Hot_Encode.fit_transform(X).toarray()
Data_Trained=X[:89594,:]
print(len(Data_Trained))
Data_Trained_o=Y.iloc[:89594,:]
print(Data_Trained_o)
model=LinearRegression()
result=model.fit(Data_Trained,Data_Trained_o)

Valid_Data=X[89594:111993,:]
Valid_Data_o=Y.iloc[89594:111993,:]
print(model.score(Valid_Data,Valid_Data_o))
Predict_data=X[111993:,:]
lis1=model.predict(predict_data)
print(lis1)

with open('new.csv','w') as f:
	csv_writer=csv.writer(f,delimiter='\n')
	csv_writer.writerow(lis1)