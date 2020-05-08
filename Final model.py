#Importing the required libraries
import statsmodels.formula.api as smf
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

array=df_train.values

X=array[:,0:4]
Y=array[:,4]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=123)

Xg=xgb.XGBRegressor(colsample_bytree=0.9,learning_rate=0.2,max_depth=7,alpha=10,n_estimators=50)

Xg.fit(X_train,Y_train)
Xg_pred=Xg.predict(X_test)

#Computing the RMSE for calculating the mean square error
rmse=np.sqrt(mean_squared_error(Y_test,Xg_pred))
###############################################################################################################
#Deploying the model
import pickle
# Saving model to disk
pickle.dump(Xg, open('FMCG.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('FMCG.pkl','rb'))

score = model.score(X_test, Y_test)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = model.predict(X_test)

##########################################################################################################
#Visualizing the original and predicted data in a plot    
x_ax = range(len(Y_test))
plt.scatter(x_ax, Y_test, s=5, color="blue", label="original")
plt.plot(x_ax, Xg_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()









