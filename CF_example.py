#!/usr/bin/env python
# coding: utf-8

# In[25]:


'''
parts of code referenced from https://james-brennan.github.io/posts/counterfactuals/
original dataset from https://www.kaggle.com/datasets/granjithkumar/loan-approval-data-set?resource=download
'''

#import tools
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

#load data
loan_data = pd.read_csv("Loan_data_clean.csv",header = 0, index_col = 0)

print(loan_data.head())


# In[37]:


#separate feature values (x) from outcomes (y)
X_values = loan_data.loc[:,"Gender":"Property_Area"]
y_values = loan_data["Loan_Status"]

print(X_values.head())
print(y_values.head())

#create and fit a model
classifier = SVC(probability = True)
classifier.fit(X_values,y_values)


# In[38]:


import scipy.stats 
#define loss/cost function
def cost_function(x, x_prime, y_prime, lambda_value, model, X):
    mad =  scipy.stats.median_abs_deviation(X, axis=0)
    distance = np.sum(np.abs(x-x_prime)/mad)
    misfit = (model(x_prime, y_prime)-y_prime)**2
    return lambda_value * misfit + distance


# In[41]:


#define function to evaluate model/predict probability of getting a target value y'
def evaluate_model(x, y_prime):
    # round the y_prime value to provide the right class [0,1]
    predicted_prob = classifier.predict_proba(np.array(x).reshape((1, -1)))[0,int(np.round(y_prime))]
    return predicted_prob


# In[86]:


#get a counterfacutal using algorithm described in book
import random

print("getting counterfactuals...")

#Select an instance x to be explained, the desired outcome y’, a tolerance ϵ and a (low) initial value for λ
instance_x = X_values.loc[1,:]
y_prime_target = 1
eps = 0 # tolerance
lambda_val = 1e-5

#Sample a random instance as initial counterfactual
sampled_indexes = [1]
random_indx = random.choice(X_values.index)
while random_indx in sampled_indexes:
    random_indx = random.choice(X_values.index)
sampled_indexes.append(random_indx)
initial_cfg = X_values.loc[random_indx]

arguments = instance_x, y_prime_target, lambda_val, evaluate_model, X_values
results = scipy.optimize.minimize(cost_function, instance_x, args = arguments)
x_prime_hat = results.x
candidates = [x_prime_hat]
y_primes = [evaluate_model(x_prime_hat,y_prime_target)]

#While |^f(x′)−y′|>ϵ: 
x_prime_prediction = evaluate_model(initial_cfg, y_prime_target)
iterations = 0
while np.abs(x_prime_prediction - y_prime_target) != eps:
    #Increase λ 
    lambda_val += 2e-10
    #Optimize the loss with the current counterfactual as starting point.
    arguments = x_prime_hat, y_prime_target, lambda_val, evaluate_model, X_values
    results = scipy.optimize.minimize(cost_function, instance_x, args = arguments)
    #Return the counterfactual that minimizes the loss.
    x_prime_hat = results.x
    x_prime_prediction = evaluate_model(x_prime_hat, y_prime_target)
candidates = x_prime_hat
y_primes = evaluate_model(x_prime_hat,y_prime_target)


print("Done")
    


# In[87]:


#print results
print(candidates, y_primes)


# In[ ]:




