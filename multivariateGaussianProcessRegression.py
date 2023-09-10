#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:25:57 2023

@author: rodneygreene
"""

import datetime as dt
import pandas as pd
import numpy as np
import fredpy as fp  # See https://www.briancjenkins.com/fredpy/docs/build/html/index.html
import matplotlib.pyplot as plt
import copy

#Packages for Gaussian Process regression
import torch
import gpytorch
from gpytorch.kernels import  PeriodicKernel, CosineKernel, LinearKernel, RBFKernel, RQKernel, SpectralMixtureKernel, ScaleKernel
import gc
from sklearn.metrics import mean_squared_error
from scipy import signal  #For power spectral density plots


#Class implementation of the Gaussian Process prior. This model is used for all kernels except the spectral
#mixture kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    
    #Returns the GP prior. Allows for non-zero mean but usually the zero mea assumption is satisfactory. Recall
    # the von Mises conditioning on this prior (which is done by the gpytorch.likelihood object)
    #to obtain the predictive posterior obtains a non-zero mean
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



#Class implementation of the Gaussian Process prior for the spectral
#mixture kernel. Note the call to initialize_from_data, which initializes the
#kernel based on the training data observations. This implementation is sourced from 
# https://docs.gpytorch.ai/en/latest/examples/01_Exact_GPs/Spectral_Mixture_GP_Regression.html
#An additional input argument, the Kernel, is added here.
class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
class GPModel:
    def __init__(self, argsDictionary:dict) -> None:
        fp.api_key = argsDictionary['FRED_APIkey']
        fredSymbolList = argsDictionary['ListOfFREDsymbols']
        
        #Define the domain for the training data observations
        self.m_fitDomainStartDate = argsDictionary['FitDomainStartDate']
        self.m_fitDomainEndDate = argsDictionary['FitDomainEndDate']
        #Define the domain for the test data
        self.m_testDomainStartDate = argsDictionary['TestDomainStartDate']
        self.m_testDomainEndDate = argsDictionary['TestDomainEndDate']
        
        self.mDaysPerYear = argsDictionary['DaysPerYear']

        #The input argsDictionary['ListOfFREDsymbols'] is a list of conomic timeseries to be sourced from FRED.
        #The dates on the first timeseries defines the anchor dates - the remaining timeseries will be date-aligned
        #to this anchor series via a merge function call.
        listIndex = -1
        for fredSymbol in fredSymbolList:
            listIndex += 1
            theSeries = fp.series(fredSymbol)
            theDF = pd.DataFrame( {fredSymbol: theSeries.data} )
            theDF.index = pd.to_datetime(theDF.index).date
            theDF.sort_index(ascending=True, inplace=True)
         
            if(listIndex == 0):
                self.mDataDF = theDF
            else:
                self.mDataDF = pd.merge(self.mDataDF, theDF, how='inner', left_on=None, right_on=None, left_index=True, right_index=True)
          
        #Unfortunately, the fredpy API does not allow the date range to be specified. So just re-set the fit domain start date to be the 
        #first date in the series 
        if( self.m_fitDomainStartDate < self.mDataDF.index[0] ):
             self.m_fitDomainStartDate = self.mDataDF.index[0]
        
        #Filter to get all the data from the start of the training data to the end of the test data domain.
        self.mDataDF = copy.deepcopy( self.mDataDF.loc[ (self.mDataDF.index >= self.m_fitDomainStartDate) &  (self.mDataDF.index <= self.m_testDomainEndDate)] )
        
        #Quick cleaning in-case there are any NaN elements
        self.mDataDF.dropna(axis=0, inplace=True)
        
        #Add a column that reflects the years to the timeseries observation date (the index of the dataframe)
        #from the start of the training domain. Recall most timeseries are displayed as a function of time.            
        tenorList = [] 
        for rowIndex in range(0, self.mDataDF.shape[0]):
            tenor = self.mDataDF.index[rowIndex] - self.mDataDF.index[0]
            tenorInYears = tenor.days / self.mDaysPerYear
            tenorList.append(tenorInYears)
        self.mDataDF['TenorInYears'] = tenorList
        
        #Simple cleaning to insure all the elements of the dataframe are type float
        for columnName in self.mDataDF.columns:
            self.mDataDF = self.mDataDF.astype( {columnName : float }) 
        
        #It's convenient to create member variables that contain the training data and the test data
        self.mData_fitDomainDF = copy.deepcopy( self.mDataDF.loc[ (self.mDataDF.index >= self.m_fitDomainStartDate) &  (self.mDataDF.index <= self.m_fitDomainEndDate)] )
        self.mData_testDomainDF = copy.deepcopy( self.mDataDF.loc[ (self.mDataDF.index >= self.m_testDomainStartDate) &  (self.mDataDF.index <= self.m_testDomainEndDate)] )


    def plotFredsymbols(self, columnList)-> None:
        for columnName in columnList:
            plt.plot(self.mData_fitDomainDF.index, self.mData_fitDomainDF[columnName], 'k-')
            plt.title(columnName)
            plt.xticks(rotation = 25)
            filename = 'fitDomain_' + columnName + '.png'
            plt.savefig(filename)
            plt.show()
            
            # plt.plot(self.mDataDF.index, self.mDataDF[columnName], 'k-')
            # plt.axvline(x = self.m_fitDomainStartDate, color = 'b', linestyle='--')
            # plt.axvline(x = self.m_fitDomainEndDate, color = 'b', linestyle='--')
            # plt.axvline(x = self.m_testDomainStartDate, color = 'g', linestyle='dashdot')
            # plt.axvline(x = self.m_testDomainEndDate, color = 'g', linestyle='dashdot')
            # plt.title(columnName)
            # plt.xticks(rotation = 25)
            # plt.show()
                    
            # fs = self.mDaysPerYear
            # frequencies, powerSprectralDensity = signal.welch(self.mDataDF[columnName].to_numpy(), fs)
            # plt.plot(frequencies[0:21], powerSprectralDensity[0:21], 'k-')  #Looks like just 21 frequencies is enough
            # plt.title('Power Spectral Density: ' + columnName) 
            # plt.xlabel('Frequency [1/year]')
            # plt.show()


    #responseColumn is a string that is the FRED symbol of the variable that is a function of the drivers
    #driverColumnlList is a list of the FRED symbols that are ordinates that drive the response variable
    def createTrainAndTestData(self, driverColumnlList:list(), responseColumn:str())->None:
        # creating tensors as required by gpytorch
        self.mTrainResponseTensor = torch.tensor(self.mData_fitDomainDF[responseColumn].to_numpy(copy=True))  #yTrain
        self.mTrainDriverTensor = torch.tensor(self.mData_fitDomainDF[driverColumnlList].to_numpy(copy=True)) #xTrain 
        self.mTestResponseTensor = torch.tensor(self.mData_testDomainDF[responseColumn].to_numpy(copy=True))  #yTest
        self.mTestDriverTensor = torch.tensor(self.mData_testDomainDF[driverColumnlList].to_numpy(copy=True)) #xTest
        
        #Tensors that just contain the observation date tenors,i.e., the years to the observation date
        #with respect to the first date in the training domain.
        self.mTrainTenorTensor = torch.tensor(self.mData_fitDomainDF['TenorInYears'].to_numpy(copy=True))
        self.mTestTenorTensor = torch.tensor(self.mData_testDomainDF['TenorInYears'].to_numpy(copy=True))
        
    
    #Training a GP regression system involves maximizing the marginal likelihood function
    #with respect to the kernel parameters (lengthscale and overall scale). In order to simplify/clarify
    #the code, this optimization is not performed in this code. In this code, the kernel hyperparameters specified in
    #the input argsDictionary are used as-is.
    def forecastWithoutHyperparamOptimization_RBF(self, argsDictionary:dict)->tuple:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=argsDictionary['Likelihood Noise'])
        
        #Create the train and test tensors
        self.createTrainAndTestData(argsDictionary['DriverColumnList'], argsDictionary['ResponseColumn'])
        
        #The squared-exponential kernel is the standard for GP regression
        #ard_num_dims = 1 for standard univariate timeseries. ard_num_dims > 1 for
        #multivariate regression
        theRBF = RBFKernel(ard_num_dims=len(argsDictionary['DriverColumnList'])) 
        lengthscaleVector = np.array(argsDictionary['RBFlengthscaleList'])  
        lengthscaleVector = np.reshape(lengthscaleVector, (-1, 1)) #required by gPyTorch
        theRBF.lengthscale = torch.tensor(lengthscaleVector) #Each driver variable can have a different lengthscale
        theScaledRBFKernel = ScaleKernel(theRBF)
        theScaledRBFKernel.outputscale = argsDictionary['OverallRBFscaleFactor']
        
        model = ExactGPModel(self.mTrainDriverTensor, self.mTrainResponseTensor, likelihood, theScaledRBFKernel.double() )
        
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():            
            #Use the predictive posterior to calculate the regression predictions over the training domain            
            gprPredictionsOverTrainingDomain = likelihood(model(self.mTrainDriverTensor))
            
            #Calculate the MSE of the GPR predictions
            mseOverTrainingDomain = mean_squared_error(self.mTrainResponseTensor.cpu().numpy(), gprPredictionsOverTrainingDomain.mean.cpu().numpy())
                
            #Calculate the power spectral density. The sampling period is daily, hence the sampling frequency is self.mDaysPerYear          
            samplingFrequency = self.mDaysPerYear
            gprPredictionFrequenciesOverTrainingDomain, gprPredictionPowerSpectralDensityOverTrainingDomain = \
                signal.welch(gprPredictionsOverTrainingDomain.mean.cpu().numpy(), samplingFrequency)

        return( gprPredictionsOverTrainingDomain, mseOverTrainingDomain,\
                gprPredictionFrequenciesOverTrainingDomain, gprPredictionPowerSpectralDensityOverTrainingDomain)


    #See the comments at the top of forecastWithoutHyperparamOptimization_RBF. The difference here is the GP-prior is 
    #SpectralMixtureGPModel. 
    def forecastWithoutHyperparamOptimization_SpectralMixtureKernel(self, argsDictionary:dict)->tuple:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=argsDictionary['Likelihood Noise'])
        
        #Create the train and test tensors
        self.createTrainAndTestData(argsDictionary['DriverColumnList'], argsDictionary['ResponseColumn'])
        
        #num_mixtures = number of kernel components. Typically <= 10.
        #ard_num_dims = 1 for standard univariate timeseries. ard_num_dims > 1 for
        #multivariate regression
        theSMkernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=argsDictionary['NumSpectralMixtureComponents'], 
                                                             ard_num_dims=len(argsDictionary['DriverColumnList']))
                
        model = SpectralMixtureGPModel(self.mTrainDriverTensor, self.mTrainResponseTensor, likelihood, theSMkernel.double() )
        
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #Use the predictive posterior to calculate the regression predictions over the training domain            
            gprPredictionsOverTrainingDomain = likelihood(model(self.mTrainDriverTensor))
            
            #Calculate the MSE of the GPR predictions
            mseOverTrainingDomain = mean_squared_error(self.mTrainResponseTensor.cpu().numpy(), gprPredictionsOverTrainingDomain.mean.cpu().numpy())
                
            #Calculate the power spectral density. The sampling period is daily, hence the sampling frequency is self.mDaysPerYear          
            samplingFrequency = self.mDaysPerYear
            gprPredictionFrequenciesOverTrainingDomain, gprPredictionPowerSpectralDensityOverTrainingDomain = \
                signal.welch(gprPredictionsOverTrainingDomain.mean.cpu().numpy(), samplingFrequency)

        return( gprPredictionsOverTrainingDomain, mseOverTrainingDomain,\
                gprPredictionFrequenciesOverTrainingDomain, gprPredictionPowerSpectralDensityOverTrainingDomain)
            
            
    def plotResults(self, argsDictionary)-> None:            
            plt.plot(self.mTrainTenorTensor.cpu().numpy(), self.mTrainResponseTensor.cpu().numpy(), 'k.', markersize=3, label='Historical Data')
            plt.plot(self.mTrainTenorTensor.cpu().numpy(), argsDictionary['Tensor GP Posterior Prediction Over Training Domain'].cpu().numpy(), 'g+', markersize=1, label='GP Posterior Prediction')
            plt.legend()
            theTitle = 'GP Prediction: ' + argsDictionary['ResponseColumn'] + '(' + str(argsDictionary['DriverColumnList']) + '): ' + argsDictionary['Kernel Name']
            plt.title(theTitle)
            plt.xlabel('Tenor [Years]')           
            filename = 'gpPrediction_' + argsDictionary['ResponseColumn'] + '_' + argsDictionary['Kernel Name'] + '.png'
            plt.savefig(filename)
            plt.show()
            
            samplingFrequency = self.mDaysPerYear
            #Get the power spectral density of the training data
            trainDataFrequencies, trainDataPowerSpectralDensity = signal.welch(self.mTrainResponseTensor.cpu().numpy(), samplingFrequency)
            plt.plot(trainDataFrequencies[0:21], trainDataPowerSpectralDensity[0:21], 'k-', label='PSD(Historical Data)')               
            plt.plot(argsDictionary['PSD Frequencies'][0:21], argsDictionary['Power Spectral Density'][0:21], 'g--', label='PSD(GP Posterior Prediction)')
            plt.xlabel('Frequency')
            theTitle = 'PSD: ' + argsDictionary['ResponseColumn'] + '(' + str(argsDictionary['DriverColumnList']) + '): ' + argsDictionary['Kernel Name']
            plt.title(theTitle)
            plt.legend()            
            filename = 'psd_' + argsDictionary['ResponseColumn'] + '_' + argsDictionary['Kernel Name'] + '.png'
            plt.savefig(filename)
            plt.show()