#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:52:18 2023

@author: rodneygreene
"""

from multivariateGaussianProcessRegression import *


argsDictionary = { 'ListOfFREDsymbols': ['SP500', 'VIXCLS', 'DFF', 'DGS10'],\
                   'FitDomainStartDate': dt.date(2021, 3, 1), 'FitDomainEndDate': dt.date(2023, 6, 30),\
                   'TestDomainStartDate':  dt.date(2023, 7, 3),\
                   'TestDomainEndDate': dt.date(2023, 7, 31),\
                   'ResponseColumn': 'SP500',\
                   'DaysPerYear': 250.0,\
                   'Likelihood Noise': 0.1**2}

#Getting the API key is free and easy - go to 
# https://fred.stlouisfed.org/docs/api/api_key.html
argsDictionary['FRED_APIkey'] = '1b22cc7ba510da2a47f1c68e41995cef'

#Instantiate the GPModel and plot the downloaded timeseries
theGPModel = GPModel(argsDictionary) 
theGPModel.plotFredsymbols( ['SP500', 'VIXCLS', 'DFF', 'DGS10'] )   

#Note that this code does not perform hyperparameter optimization. The hyperparameters used here
#were determined via trial and error

#--- SP500( tenor ), the simplest univariate timeseries
argsDictionary['DriverColumnList'] = ['TenorInYears']
#Hyperparameters for the squared exponential kernel. 
argsDictionary['OverallRBFscaleFactor'] = 50.0**2
argsDictionary['RBFlengthscaleList'] = [11.0]
argsDictionary['NumSpectralMixtureComponents'] = 5

rbfResultsTuple = theGPModel.forecastWithoutHyperparamOptimization_RBF(argsDictionary)
argsDictionary['Kernel Name'] = 'RBF'
argsDictionary['Tensor GP Posterior Prediction Over Training Domain'] = rbfResultsTuple[0].mean
argsDictionary['PSD Frequencies'] = rbfResultsTuple[2]
argsDictionary['Power Spectral Density'] = rbfResultsTuple[3]
theGPModel.plotResults(argsDictionary)

smResultsTuple = theGPModel.forecastWithoutHyperparamOptimization_SpectralMixtureKernel(argsDictionary)
argsDictionary['Kernel Name'] = 'SM'
argsDictionary['Tensor GP Posterior Prediction Over Training Domain'] = smResultsTuple[0].mean
argsDictionary['PSD Frequencies'] = smResultsTuple[2]
argsDictionary['Power Spectral Density'] = smResultsTuple[3]
theGPModel.plotResults(argsDictionary)

print('Drivers:' + str(argsDictionary['DriverColumnList']) + '\t RBF MSE = ', rbfResultsTuple[1], '\t SM MSE = ', smResultsTuple[1])

#
#--- SP500( tenor, VIX ), a multivariate GP regression
#
argsDictionary['DriverColumnList'] = ['TenorInYears', 'VIXCLS']
#Hyperparameters for the squared exponential kernel. 
argsDictionary['OverallRBFscaleFactor'] = 50.0**2
argsDictionary['RBFlengthscaleList'] = [11.0, 10.5]
argsDictionary['NumSpectralMixtureComponents'] = 5

rbfResultsTuple = theGPModel.forecastWithoutHyperparamOptimization_RBF(argsDictionary)
argsDictionary['Kernel Name'] = 'RBF'
argsDictionary['Tensor GP Posterior Prediction Over Training Domain'] = rbfResultsTuple[0].mean
argsDictionary['PSD Frequencies'] = rbfResultsTuple[2]
argsDictionary['Power Spectral Density'] = rbfResultsTuple[3]
theGPModel.plotResults(argsDictionary)

smResultsTuple = theGPModel.forecastWithoutHyperparamOptimization_SpectralMixtureKernel(argsDictionary)
argsDictionary['Kernel Name'] = 'SM'
argsDictionary['Tensor GP Posterior Prediction Over Training Domain'] = smResultsTuple[0].mean
argsDictionary['PSD Frequencies'] = smResultsTuple[2]
argsDictionary['Power Spectral Density'] = smResultsTuple[3]
theGPModel.plotResults(argsDictionary)

print('Drivers:' + str(argsDictionary['DriverColumnList']) + '\t RBF MSE = ', rbfResultsTuple[1], '\t SM MSE = ', smResultsTuple[1])

#
#--- SP500( tenor, VIX, Fed Fund Effective Rate ), a multivariate GP regression
#
argsDictionary['DriverColumnList'] = ['TenorInYears', 'VIXCLS', 'DFF']
#Hyperparameters for the squared exponential kernel. 
argsDictionary['OverallRBFscaleFactor'] = 50.0**2
argsDictionary['RBFlengthscaleList'] = [11.0, 10.5, 8.0]
argsDictionary['NumSpectralMixtureComponents'] = 5

rbfResultsTuple = theGPModel.forecastWithoutHyperparamOptimization_RBF(argsDictionary)
argsDictionary['Kernel Name'] = 'RBF'
argsDictionary['Tensor GP Posterior Prediction Over Training Domain'] = rbfResultsTuple[0].mean
argsDictionary['PSD Frequencies'] = rbfResultsTuple[2]
argsDictionary['Power Spectral Density'] = rbfResultsTuple[3]
theGPModel.plotResults(argsDictionary)

smResultsTuple = theGPModel.forecastWithoutHyperparamOptimization_SpectralMixtureKernel(argsDictionary)
argsDictionary['Kernel Name'] = 'SM'
argsDictionary['Tensor GP Posterior Prediction Over Training Domain'] = smResultsTuple[0].mean
argsDictionary['PSD Frequencies'] = smResultsTuple[2]
argsDictionary['Power Spectral Density'] = smResultsTuple[3]
theGPModel.plotResults(argsDictionary)

print('Drivers:' + str(argsDictionary['DriverColumnList']) + '\t RBF MSE = ', rbfResultsTuple[1], '\t SM MSE = ', smResultsTuple[1])

