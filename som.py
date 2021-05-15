# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:10:29 2021

@author: rusal
"""

#Importing all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
data = pd.read_csv('Credit_Card_Applications.csv')

#Separating data for Unsupervised learning
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#Feature Scaling our data
from sklearn.preprocessing import MinMaxScaler
scaled_data = MinMaxScaler(feature_range = (0,1))
x = scaled_data.fit_transform(x)


#Importing MiniSom for preparing our model
from minisom import MiniSom
som_model = MiniSom(x=10,y=10,input_len = 15, sigma = 1.0,learning_rate = 0.5)

#Initialising the weights for our model using the functions present in MiniSom
som_model.random_weights_init(x)
som_model.train_random(data = x, num_iteration = 100)

#Importing libaries for plotting our Map
from pylab import bone, pcolor, colorbar, show, plot

#Plotting customers with and without approval using markers and colours
bone()
pcolor(som_model.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['b','r']

for i,z in enumerate(x):
    v = som_model.winner(z)
    plot(v[0] + 0.5,
         v[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markeredgewidth = 1,
         markersize = 9
         )
    
show()

#Finding out potenetially fraud customers using the visualisations from the plot and the corresponding coordinates

mappings = som_model.win_map(x)

fraud_cust = np.concatenate((mappings[(6,3)], mappings[(7,1)]), axis = 0)

#The data we get in our list is our scaled data, so we will now apply inverse_transform to retreive the orginial data of these fraud customers
fraud_cust = scaled_data.inverse_transform(fraud_cust)

#This now gives us a list of all the potential fraud customers from the given data