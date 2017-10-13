#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:47:44 2017

@author: mac
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

def get_file(file_name, x_step_size, x_end_point,y_step_size,y_end_point):
    
    if x_end_point - (x_end_point/x_step_size*x_step_size) == 0:
        x_axis = int(x_end_point/x_step_size) +1
    else:
        x_axis = int(x_end_point/x_step_size) +2


    if y_end_point -(y_end_point/y_step_size*y_step_size) ==0:
        y_axis = int(y_end_point/y_step_size) +1
    else:
        y_axis = int(y_end_point/y_step_size) +2  

    file_name_full = file_name+'.csv'
    sp = int(file_name[-1])
    
    sample = np.loadtxt(file_name_full, delimiter=',')

    inten = sample.shape[0]

    z = np.zeros((int(x_axis)*int(y_axis),inten))

    for i in range(sp,int(x_axis*y_axis)+sp):
        file_name_full = file_name[:-1]+str(i)+'.csv'
        data = np.loadtxt(file_name_full, delimiter=',')
        z[i-sp] = data.transpose()[1]

    z = np.flip(z,0)#그대로 
    return z, x_axis, y_axis, sample

def td_scan_plot(k1,k2,x_pos,y_pos,x_step_size,y_step_size,x_axis,y_axis,sample,z):
    import plotly 
    plotly.tools.set_credentials_file(username='andychoi153', api_key='jPgMrSckVeZctbKO9Meo')

    import plotly.plotly as py
    import plotly.graph_objs as go
    from plotly import tools


    pos = int(y_pos/y_step_size)*int(x_axis)+ (x_axis-(int(x_pos/x_step_size)+1))# 이걸 참조하세요!!

    x = np.arange(0, x_axis*x_step_size, x_step_size)
    y = np.arange(0,  y_axis*y_step_size, y_step_size)

    sample_int = (sample.transpose()[0]*10).astype(int)
    i = np.argwhere(sample_int==int(k1*10))

    data = np.flip(z.transpose()[i].reshape((int(x_axis) ,int(y_axis))),1)

    trace = go.Heatmap(z=data,x = x, y= y)
    data=[trace]

    trace0 = go.Scatter(x = sample.transpose()[0],y = z[pos])
    sample_int = (sample.transpose()[0]*10).astype(int)
    i = np.argwhere(sample_int==int(k2*10))

    data = np.flip(z.transpose()[i].reshape((int(x_axis) ,int(y_axis))),1)

    trace1 = go.Heatmap(z=data,x = x, y= y)

    fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                              subplot_titles=('Scan '+str(k1)+'nm','Scan '+str(k2)+'nm', 'Intensity'))

    fig.append_trace(trace, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace0, 2, 1)

    fig['layout'].update(height=800, width=800, title='2D Scan')
    return py.iplot(fig, filename='basic-heatmap')#그대로
    
def x_wavelength(sample):
    x = sample.transpose()[0]
    return x

def get_wavelength_index(sample, wavelength, z):
    x = x_wavelength(sample)
    sample_int = (x*10).astype(int)
    i = np.argwhere(sample_int==int(wavelength*10))
    y = z.transpose()[i]
    print(y[0][0])
    print(y.shape)
    index_peak = find_peak(y[0][0])
    return index_peak

def transfer_wavelength(start_wl,end_wl,sample):
    i = np.argwhere(sample > start_wl)
    start =  i[0]
    i = np.argwhere(sample < end_wl)
    end = i[-1]
    return start, end

def plot_Peak_image(index, z, start_wl, end_wl, x):
    start_wl, end_wl = transfer_wavelength(start_wl, end_wl)
    
    for i in index:
        plt.plot(x[start_wl:end_wl],z[i][start_wl:end_wl])
        
def find_peak(y):# input 은 scan image 소스 - 일렬 배열
    from scipy import signal
    peakind = signal.find_peaks_cwt(y, np.arange(1,1000,500))
    #return 값은 해당 소스의 peak index를 가져온다.
    return peakind

def resize_wavelength(start_wl, end_wl, z, sample):
    start_wl, end_wl = transfer_wavelength(start_wl, end_wl, sample)
    z_new = z.transpose()[start_wl:end_wl].transpose()
    return z_new

def cluster_sample(pca_z,n):
    
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(pca_z)

    #x = [np.argmax(pca_z), np.argmin(pca_z)]
    #y = [gmm.predict(pca_z)[x[0]],gmm.predict(pca_z)[x[1]]]
    
    return gmm.predict(pca_z)

def print_cluster(pre, z, x ,n, start, end):
    fig, axes = plt.subplots(nrows=int(n/2), ncols=2, figsize=(20, 15))
    for j in range(n):
        print(np.argwhere(pre==n).shape)
    for j in range(n):
        for i in np.argwhere(pre==j):
            axes[int(j/2), j%2].plot(x[start:end],z[i[0]][start:end])
            plt.ylim(400,1000)