#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pycaret


# In[2]:


import pandas as pd


# In[3]:


dataset=pd.read_csv('C:\\Users\\vitis\\OneDrive\\Desktop\\4 TH SEM MBA\\LIVE PROJECT\\CAD data April 2022_V2_Control_Vitish.csv')#scaled for max vlaues in excel itself

# check the shape of data
dataset.shape


# In[4]:


# sample 5% of data to be used as unseen data
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
# print the revised shape
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[5]:


pip install scikit-learn==0.23.2


# In[7]:


from pycaret.classification import *
s = setup(data = data, target = 'Type', session_id=123)


# In[8]:


best_model = compare_models()


# In[9]:


print(best_model)


# In[10]:


# check available models
models()


# In[11]:


dt = create_model('dt')


# In[12]:


# trained model object is stored in the variable 'dt'. 
print(dt)


# In[13]:


tuned_dt = tune_model(dt)


# In[14]:


# tuned model object is stored in the variable 'tuned_dt'. 
print(tuned_dt)


# In[15]:


plot_model(tuned_dt, plot = 'auc')


# In[16]:


plot_model(tuned_dt, plot = 'pr')


# In[17]:


plot_model(tuned_dt, plot='feature')


# In[18]:


plot_model(tuned_dt, plot = 'confusion_matrix')


# In[19]:


evaluate_model(tuned_dt)


# In[20]:


predict_model(tuned_dt);


# In[21]:


#Finalize Model for Deployment
# finalize rf model
final_dt = finalize_model(tuned_dt)
# print final model parameters
print(final_dt)


# In[22]:


predict_model(final_dt);


# In[23]:


unseen_predictions = predict_model(final_dt, data=data_unseen)
unseen_predictions.head()


# In[24]:


# check metric on unseen data
from pycaret.utils import check_metric
check_metric(unseen_predictions['Type'], unseen_predictions['Label'], metric = 'Accuracy')


# In[25]:


# saving the final model
save_model(final_dt,'Control Healthcare')


# In[26]:


# loading the saved model
saved_final_dt = load_model('Control Healthcare')


# In[27]:


# predict on new data
new_prediction = predict_model(saved_final_dt, data=data_unseen)
new_prediction.head()


# In[30]:


#Model interpretability
import shap
from itertools import chain
from collections import defaultdict
pd.set_option('display.max_columns', None)
shap.initjs()


# In[28]:


interpret_model(final_dt)


# In[32]:


interpret_model(final_dt,plot = 'correlation')


# In[ ]:


pip install shap


# In[34]:


sudo apt-get install python3.7

