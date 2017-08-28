#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:34:43 2017

@author: soumyadipghosh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import numpy as np
from bokeh.models import ColumnDataSource,HoverTool,Select,Legend,LegendItem,ColorBar,LinearColorMapper,TextInput
from bkcharts import Bar
from bokeh.plotting import figure, output_file, show,curdoc
from bokeh.palettes import brewer,Viridis256,Plasma256
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column,row
from bokeh.models.widgets import RadioButtonGroup
from bkcharts.attributes import cat,color
from bkcharts.operations import blend
from sklearn.preprocessing import MinMaxScaler

class Back_Train:
    
    def __init__(self):
        self.dataset_t = "Dataset1"
        self.time = 0
        self.look_back = 6
        self.ahead = 0
    
    def setparams(self,dataset,var,time,look_back):
        
        if(dataset == "Dataset3"):
            self.dataset_t = pd.read_csv("training_data_2015.csv")
        else:
            self.dataset_t = pd.read_csv("training_data_set.csv")
        self.dataset_t = self.dataset_t.dropna()
        
        if(dataset == "Dataset1"):
            self.price  = self.dataset_t.iloc[:,2:3].values
            self.load = self.dataset_t.iloc[:,3:4].values
            self.test_set = pd.read_csv("test_data_1.csv")
        elif(dataset == "Dataset2"):
            self.price  = self.dataset_t.iloc[:,4:5].values
            self.load = self.dataset_t.iloc[:,5:6].values
            self.test_set = pd.read_csv("test_data_2.csv")
        else:
            self.price  = self.dataset_t.iloc[:,3:4].values
            self.load = self.dataset_t.iloc[:,4:5].values
            self.test_set = pd.read_csv("training_data_set.csv")
        
        self.test_set.dropna()
        if(var == 1):
            self.var = self.price
            self.test_var = self.test_set.iloc[:,2:3].values
        else:
            self.var = self.load
            self.test_var = self.test_set.iloc[:,3:4].values

        self.look_back = look_back
        self.ahead = time
        

    def create_dataset(self):
        dataX, dataY = [], []
        for i in range(len(self.dataset_t)-self.look_back-self.ahead):
            dataX.append(self.dataset_t[i:(i+self.look_back), 0])
            dataY.append(self.dataset_t[i +self.look_back + self.ahead, 0])
        return np.array(dataX), np.array(dataY)

    def result(self):
        sc = MinMaxScaler()
        self.var = sc.fit_transform(self.var)
        print (self.var)
        dataX, dataY = [], []
        for i in range(len(self.var)-self.look_back-self.ahead):
            dataX.append(self.var[i:(i+self.look_back), 0])
            dataY.append(self.var[i +self.look_back + self.ahead, 0])
        X_train = np.array(dataX)
        Y_train = np.array(dataY)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
       
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        regressor = Sequential()
        #
        batch_size = 64
        ## Adding the input layer and the LSTM layer
        regressor.add(LSTM(units =12, activation = 'sigmoid', input_shape = (None,1),implementation=2,stateful=False,return_sequences=False))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.fit(X_train, Y_train, batch_size = batch_size, epochs = 3)
        
        self.test_var = self.test_var[1:] 
        print (self.test_var)
        print (type(self.test_var[0][0]))
        if(isinstance(self.test_var[0],str)):
            for i in range(len(self.test_var)):
                self.test_var[i][0] = float(self.test_var[i][0].replace(',',''))
#        
        self.test_var = self.test_var.astype(dtype='float64')
    
        dataX, dataY = [], []
        for i in range(len(self.test_var)-self.look_back-self.ahead):
            dataX.append(self.test_var[i:(i+self.look_back), 0])
            dataY.append(self.test_var[i +self.look_back + self.ahead, 0])
        X_test = np.array(dataX)
        Y_test = np.array(dataY)
        
        inputs = X_test
        inputs = inputs[~np.isnan(inputs).any(axis=1)]
        bad_indices = np.where(np.isnan(inputs))
        inputs = sc.transform(inputs)
        inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],1))
        predicted_load = regressor.predict(inputs)
        predicted_load = sc.inverse_transform(predicted_load)
        return predicted_load.ravel() , Y_test.ravel()        
        
#        color = brewer['Set1'][3]
#        output_file("energy_2.html")
#        demand_plot_1 = figure(plot_width=1000, plot_height=300,title="Price")
#        demand_plot_1.xaxis.axis_label = "Hours"
#        demand_plot_1.yaxis.axis_label = "Price(Euro)"
#        demand_plot_1.line(np.linspace(1,time,time),predicted_load[:time,].ravel(),line_width=2,legend="Predicted Price",color=color[0])
#        demand_plot_1.line(np.linspace(1,time,time),y_test[:time,].ravel(),line_width=2,legend="Real Price",color=color[1])
#        #demand_plot_1.line(np.linspace(1,8782,8782),price.ravel(),line_width=2,legend="Realy",color=color[2])
#        demand_plot_1.legend.location = "top_left"
#        demand_plot_1.legend.click_policy="hide"
#        show(demand_plot_1)

#def mape(actual, predict): 
#    tmp, n = 0.0, 0
#    for i in range(0, min(len(actual),len(predict))):
#        if actual[i] != 0:
#            tmp += math.fabs(actual[i]-predict[i])/actual[i]
#            n += 1
#            a.append(tmp)
#    return (tmp/n)
#
#print(mape(y_test,predicted_load.ravel()))


class Visual_Front:
    
    def __init__(self):
        
        self.datasets = ["Dataset1","Dataset2","Dataset3"]
        self.vars = ["Price(Euro)","Load(MW)"]
        self.times = ["Hourly","Daily","Weekly"]
        self.algorithms = ["RNN + LSTM"]
        
    def layout(self):
        
        def update_plot(attr,new,old):
            dataset = datasets_select.value
            
            if(vars_select.value == "Price(Euro)"):
                var = 1
            else:
                var = 2
            
            look_back = int(look_back_select.value)
            print(type(look_back))
            
            if(times_select.value == "Hourly"):
                time = 0
            elif(times_select.value == "Daily"):
                time = 24 - look_back
            else:
                time = 168 - look_back
            
            
#            backend = Back_Train()
            backend.setparams(dataset,var,time,look_back)
            pred,actual = backend.result()
            time = min(len(pred),len(actual))
            
            dis_plot_1 = figure(plot_width=1200, plot_height=300,title="Energy Consumption per Node")
            dis_plot_1.xaxis.axis_label = "Hours"
            dis_plot_1.yaxis.axis_label = "Value"
            dis_plot_1.legend.location = "top_left"
            dis_plot_1.legend.click_policy="hide"
            
            dis_plot_1.line(np.linspace(1,time,time),pred[:time,],line_width=2,legend="Predicted Price",color=color_p[0])
            dis_plot_1.line(np.linspace(1,time,time),actual[:time,],line_width=2,legend="Real Price",color=color_p[1])
            
            row_1.children[0] = dis_plot_1
            return
        
        datasets_select = Select(value="Dataset1", title='Datasets', options=self.datasets)
        vars_select = Select(value="Price(Euro)", title='Value', options=self.vars)
        times_select = Select(value="Hourly", title='Time Frame', options=self.times)
        algorithms_select = Select(value="RNN + LSTM", title='', options=self.algorithms)
        look_back_select = TextInput(value="6", title="Look Back")
        datasets_select.on_change('value', update_plot)
        vars_select.on_change('value', update_plot)
        times_select.on_change('value',update_plot)
        algorithms_select.on_change('value', update_plot)
        look_back_select.on_change("value", update_plot)
        
        dataset = datasets_select.value
            
        if(vars_select.value == "Price(Euro)"):
            var = 1
        else:
            var = 2
        
        look_back = int(look_back_select.value)
        print((type(look_back)))
        
        if(times_select.value == "Hourly"):
            time = 0
        elif(times_select.value == "Daily"):
            time = 24 - look_back
        else:
            time = 168 - look_back
        
        print (type(time))
        backend = Back_Train()
        backend.setparams(dataset,var,time,look_back)
        pred,actual = backend.result()
        time = min(len(pred),len(actual))
        
        dis_plot = figure(plot_width=1200, plot_height=300,title="Energy Consumption per Node")
        dis_plot.xaxis.axis_label = "Hours"
        dis_plot.yaxis.axis_label = "Value"
        dis_plot.legend.location = "top_left"
        dis_plot.legend.click_policy="hide"
        
        color_p = brewer['Set1'][3]
        dis_plot.line(np.linspace(1,time,time),pred[:time,],line_width=2,legend="Predicted Price",color=color_p[0])
        dis_plot.line(np.linspace(1,time,time),actual[:time,],line_width=2,legend="Real Price",color=color_p[1])
        
        row_1 = row(dis_plot)
        row_2 = row(datasets_select,vars_select,times_select,algorithms_select,look_back_select)
        
        return column(row_1,row_2)

viz = Visual_Front()
curdoc().add_root(viz.layout())
#show(viz.layout())   
        
        


#import statsmodels.api as sm
#
#zone = 2184
#
#df_real = pd.DataFrame(predicted_load[0:zone,])
#df_real.rename(columns={ df_real.columns[0]: "whatever" },inplace=True)
#date_index2 = pd.date_range('1/1/2016 00:00:00', periods=zone, freq='H')
#df_real.index = date_index2
#df_real.interpolate(inplace=True)
#
#res = sm.tsa.seasonal_decompose(df_real,model='additive')
#resplot = res.plot()
#
#print (res.trend)
#res.trend.plot()

#
#demand_plot_2 = figure(plot_width=1000, plot_height=300,title="Seasonal")
#demand_plot_2.xaxis.axis_label = "Time"
#demand_plot_2.yaxis.axis_label = "Seasonal(MW)"
#demand_plot_2.line(np.linspace(1,time,time),res.seasonal,line_width=2,legend="Predicted Load",color=color[0])
##demand_plot_1.line(np.linspace(1,8782,8782),price.ravel(),line_width=2,legend="Realy",color=color[2])
#demand_plot_2.legend.location = "top_left"
#demand_plot_2.legend.click_policy="hide"
#show(demand_plot_2)


#from seasonal import fit_seasons, adjust_seasons
#seasons, trend = fit_seasons(y_test)
#adjusted = adjust_seasons(s, seasons=seasons)
#residual = adjusted - trend
