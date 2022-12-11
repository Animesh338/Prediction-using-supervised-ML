#!/usr/bin/env python
# coding: utf-8

# # The Sparks foundation #December22
#    
#     DATA SCIENCE AND BUSINESS ANALYTICS INTERN
#     
#     **AUTHOR:ANIMESH DAS**
#     
#     TASK 1:Prediction using Supervised ML.
#     
#     OBJECTIVE: We have to predict the percentage score of a student based on the number of hours studied.The task has two        variables i.e the no. of hours studied and the target value is the percentage score.
#     This is solved using Linear Regression.

# In[54]:


# import warning


# In[55]:


import warnings
warnings.filterwarnings("ignore")


# In[6]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Reading the data 
# 

# In[8]:


data=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
data.head(10)


# # Description of the data

# In[10]:


data.describe()


# In[12]:


# it showws the information of the data
data.info()


# In[14]:


# Shape of the data
print(data.shape)


# # scatter plot

# In[16]:


data.plot(kind='scatter',x='Hours',y='Scores');
plt.title('Marks  Vs  Study Hours',size=20)
plt.show()


# # Calculating the correlation coefficient

# In[18]:


data.corr(method='pearson')


# In[20]:


# The variables are positively and strongly correlated as the value of correlation coefficient is close to one.


# In[22]:


#Regression plot


# In[24]:


sns.regplot(x=data['Hours'],y=data['Scores'])
plt.title('Regression plot',size=20)
plt.xlabel('Hours studied',size=12)
plt.ylabel('Marks scored',size=12)
plt.show()
print(data.corr())


# ## It is confirmed that the variables are positively correlated.

# ### Training the model
# ### Splitting the data

# In[26]:


x=data.drop(["Scores"],axis=1)
y=data["Scores"]


# In[28]:


x


# In[30]:


y


# In[32]:


# importing train_test_split method


# In[34]:


from sklearn.model_selection import train_test_split


# In[36]:


x_train,x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=0)


# In[38]:


# Model training


# In[40]:


from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(x_train,y_train)


# In[42]:


print(clf.coef_,clf.intercept_)


# In[44]:


## visual representation of the model


# In[46]:


line=clf.coef_*x+clf.intercept_
plt.title('linear regression vs trained model')
plt.scatter(x,y,color='red')
plt.xlabel('Hours studied')
plt.ylabel('Marks scored')
plt.plot(x,line)
plt.show()


# In[48]:


print(x_test)
pred_y=clf.predict(x_test)


# ## Comparing the actual value to the predicted value

# In[50]:


df=pd.DataFrame({'Actual value':y_test,'Predicted value':pred_y})
df


# In[51]:


# checking the acccuracy of train and test data


# In[52]:


print("Training Score",clf.score(x_train,y_train))
print("Testing Score",clf.score(x_test,y_test))


# # predicting the score for 9.25 hours 

# In[56]:


hours=[9.25]
test=np.array(['hours'])
test=test.reshape(-1,1)
pred=clf.predict([[9.25]])
print("No. of hours = {}".format(hours))
print("Predicted score ={}".format(pred[0]))


# # The predicted score of a student studying for 9.25 hours is 93.69173....

# In[57]:


### checking the efficiency of the model 


# In[59]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mean_square_error=mean_squared_error(y_test,pred_y[:5])
mean_abs_error=mean_absolute_error(y_test,pred_y[:5])
print("Mean square error=",mean_square_error)
print("Mean_absolute_error",mean_abs_error)


# In[ ]:




