#!/usr/bin/env python
# coding: utf-8

# ## Set up

# ### package install

# In[ ]:


#get_ipython().system(u'sudo apt-get install build-essential swig')
#get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
#get_ipython().system(u'pip install auto-sklearn')
#get_ipython().system(u'pip install pipelineprofiler # visualize the pipelines created by auto-sklearn')
#get_ipython().system(u'pip install shap')
#get_ipython().system(u'pip install --upgrade plotly')
#get_ipython().system(u'pip3 install -U scikit-learn')


# ### Packages imports

# In[1]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import autosklearn.regression


import plotly.express as px
import plotly.graph_objects as go

from joblib import dump

import shap

import datetime

import logging

import matplotlib.pyplot as plt


# ### Google Drive connection

# In[2]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# ### options and settings

# In[3]:


data_path = "/content/drive/MyDrive/Introduction2DataScience/tutorials/w2d2/data/raw/"


# In[4]:


model_path = "/content/drive/MyDrive/Introduction2DataScience/tutorials/w2d2/models/"


# In[5]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')


# In[6]:


logging.basicConfig(filename=f"{model_path}explog_{timesstr}.log", level=logging.INFO)


# Please Download the data from [this source](https://drive.google.com/file/d/1MUZrfW214Pv9p5cNjNNEEosiruIlLUXz/view?usp=sharing), and upload it on your Introduction2DataScience/data google drive folder.

# ## Loading Data and Train-Test Split

# In[7]:


df = pd.read_csv(f'{data_path}winequality-red.csv')


# In[8]:


test_size = 0.2
random_state = 0


# In[9]:


train, test = train_test_split(df, test_size=test_size, random_state=random_state)


# In[10]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# In[11]:


train.to_csv(f'{data_path}winequality-red.csv', index=False)


# In[12]:


train= train.copy()


# In[13]:


test.to_csv(f'{data_path}winequality-red.csv', index=False)


# In[14]:


test = test.copy()


# ## Modelling

# In[15]:


X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1] 


# In[16]:


total_time = 600
per_run_time_limit = 30


# In[17]:


automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=total_time,
    per_run_time_limit=per_run_time_limit,
)
automl.fit(X_train, y_train)


# In[18]:


logging.info(f'Ran autosklearn regressor for a total time of {total_time} seconds, with a maximum of {per_run_time_limit} seconds per model run')


# In[19]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[20]:


logging.info(f'Saved regressor model at {model_path}model{timesstr}.pkl ')


# In[21]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# In[ ]:


#profiler_data= PipelineProfiler.import_autosklearn(automl)
#PipelineProfiler.plot_pipeline_matrix(profiler_data)


# ## Model Evluation and Explainability

# Let's separate our test dataframe into a feature variable (X_test), and a target variable (y_test):

# In[22]:


X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]


# #### Model Evaluation

# Now, we can attempt to predict the median house value from our test set. To do that, we just use the .predict method on the object "automl" that we created and trained in the last sections:

# In[ ]:


y_pred = automl.predict(X_test)


# Let's now evaluate it using the mean_squared_error function from scikit learn:

# In[ ]:


logging.info(f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test, y_test)}")


# we can also plot the y_test vs y_pred scatter:

# In[ ]:


df = pd.DataFrame(np.concatenate((X_test, y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))


# In[ ]:


df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'Actual Target', 'Predicted Target']


# In[ ]:


fig = px.scatter(df, x='Predicted Target', y='Actual Target')
fig.write_html(f"{model_path}residualfig_{timesstr}.html")


# In[ ]:


logging.info(f"Figure of residuals saved as {model_path}residualfig_{timesstr}.html")


# #### Model Explainability

# In[ ]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test.iloc[:50, :], link = "identity")


# In[ ]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
#shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test.iloc[X_idx:X_idx+1,:], 
                show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}shap_example_{timesstr}.png")
logging.info(f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


# In[ ]:


shap_values = explainer.shap_values(X = X_test.iloc[0:50,:], nsamples = 100)


# In[ ]:


# print the JS visualization code to the notebook
#shap.initjs()
fig = shap.summary_plot(shap_values = shap_values,
                  features = X_test.iloc[0:50,:],
                  show=False)
plt.savefig(f"{model_path}shap_summary_{timesstr}.png")
logging.info(f"Shapley summary saved as {model_path}shap_summary_{timesstr}.png")

