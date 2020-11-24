
# coding: utf-8

# In[1]:


#Datamanipulation phase 1
#import libararies
#Customer churn telecom imdustry,Customer_churn Dataset


# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[3]:


customer_churn = pd.read_csv(r'C:\Users\hanuman\Downloads\Capstone-project-2\Dataset\customer_churn.csv')


# In[4]:


customer_churn.head()


# In[8]:


#A)Data Manipulation:
#a.Extract the 5th column & store it in ‘customer_5’


# In[5]:


c_5=customer_churn.iloc[:,4]
c_5.head()


# In[7]:


#b.Extract the 15th column & store it in ‘customer_15’


# In[6]:


c_15=customer_churn.iloc[:,14]
c_15.head()


# In[11]:


#c.Extract all the male senior citizens whose Payment Method is Electronic check & store the result in ‘senior_male_electronic’


# In[7]:


c_random = customer_churn[(customer_churn['gender']=="Male") & (customer_churn['SeniorCitizen']==1) & (customer_churn['PaymentMethod']=="Electronic check")]


# In[8]:


c_random.head()


# In[14]:


#d.Extract all those customers whose tenure is greater than 70 months or their Monthly charges is more than 100$ & store the result in ‘customer_total_tenure’


# In[ ]:


#notice this is the or condition


# In[9]:


c_random =customer_churn[(customer_churn['tenure']>70) |(customer_churn['MonthlyCharges']>100)]


# In[10]:


c_random.head()


# In[19]:


#e.Extract all the customers whose Contract is of two years, payment method is Mailed check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’


# In[11]:


c_random =customer_churn[(customer_churn['Contract']=="Two Year") & (customer_churn['PaymentMethod'] =="Mailed check") & (customer_churn['Churn'] =="yes")]


# In[12]:


c_random


# In[54]:


#f.Extract 333 random records from the customer_churn dataframe & store the result in ‘customer_333’


# In[ ]:


#use the sample function,random sampling,sample method


# In[13]:


c_333 = customer_churn.sample(n=333)


# In[14]:


c_333.head()


# In[58]:


#g.Get the count of different levels from the ‘Churn’ column


# In[15]:


customer_churn['Contract'].value_counts()


# In[17]:


#B)Data Visualization:
#a.Build a bar-plot for the ’InternetService’ column:
#x-axis label to ‘Categories of Internet Service’
#ii.Set y-axis label to ‘Count of Categories’
#iii.Set the title of plot to be ‘Distribution of Internet Service’
#iv.Set the color of the bars to be ‘orange’e


# In[16]:


plt.bar(customer_churn['InternetService'].value_counts().keys().tolist(),customer_churn['InternetService'].value_counts().tolist(),color="orange")

plt.xlabel("Categories of Internet Service")
plt.ylabel("Count")
plt.title("Distribution of Internet Service")
plt.show()


# In[22]:





# In[23]:





# In[ ]:


#Histogram


# In[17]:


plt.hist(customer_churn['tenure'],bins=30,color="green")
plt.title("Distribution of tenure")
plt.show()


# In[42]:


#scatter-plot


# In[18]:


plt.scatter(x=customer_churn['tenure'],y=customer_churn['MonthlyCharges'],color='brown')
plt.xlabel('tenure')
plt.ylabel('MonthlyCharges')
plt.title('Tenure vs Monthly Charges')
plt.show()


# In[47]:


#box-plot


# In[19]:


customer_churn.boxplot(column=['tenure'],by=['Contract'])
plt.show()


# In[51]:


#Machine learning Linear Regression


# In[20]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

y=customer_churn[['MonthlyCharges']]
x=customer_churn[['tenure']]


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)


# In[22]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[23]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[28]:


y_pred=regressor.predict(x_test)
y_pred[:4],y_test[:4]


# In[29]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))

