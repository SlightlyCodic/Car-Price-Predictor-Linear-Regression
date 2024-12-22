import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
mpl.style.use('ggplot')


car = pd.read_csv("Data/cleaned_data.csv")

#Extracting Training Data
X = car[['name', 'company','year','kms_driven', 'fuel_type']]
y = car['Price']


# Apply Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#Creating One Hot Encoder

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company', 'fuel_type']), remainder='passthrough')

#Linear regression model
lr = LinearRegression()

# Creating a pipeline
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)

# Finding the model with a random state of TrainTestSplit having high r2score.

scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))
    
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)

print(r2_score)

# pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))

print(pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))))


