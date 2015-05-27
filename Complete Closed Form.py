from __future__ import division

__author__ = 'Volos'
__copyright__ = '2015 Volos Portfolio Solutions. All rights reserved'
__credits__ = ['Austin Boesch','Dan Corcoran', 'Saif Sultan','Espen Huag']
__license__ = 'EULA'
__version__ = '1.0 Python 2.7'
__maintainer__ = 'Austin Boesch'
__email__ = 'austin.boesch@volossoftware.com'
__status__ = 'Production'

#######################################Description#######################################
#This contains Closed Form Solutions
#######################################Description#######################################

####################################################Python Modules Import ################################################
import copy
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import json
import math as mth
import time as time

####################################################Python Modules Import ################################################


#start timer
start_time = time.time()


class ClosedFormPricing:
    
    def __init__(self, jsondata):
        self.Option_Type = jsondata["Option_Type"]
        self.r = self.jsondata["Interest_Rate"]
        self.T = self.jsondata["Expiration"]
        self.y = self.jsondata["Dividend_Rate"]
        self.b = self.r - self.y
        self.S0 = self.jsondata["Asset_Price"]
        self.sigma = self.jsondata["Volatility"]
          
    
    
    def getPrice(self):
        getattr(self, Option_Type)()


    def Vanilla_European_Call(self): #Validated
        self.K = self.jsondata["Single_Strike"]
        self.Vanilla_d1 = ((np.log(self.S0/self.K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))
        self.value = (self.S0 *np.exp(-self.y * self.T)* ss.norm.cdf(self.Vanilla_d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.Vanilla_d2))
        print  self.value
        return self.value

    def Vanilla_European_Put(self): #Validated
        #S0, K, r, sigma, T, y
        
        self.K = self.jsondata["Single_Strike"]

        self.Vanilla_d1 = ((np.log(self.S0/self.K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))
        self.value = (self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-self.Vanilla_d2))-(self.S0 *np.exp(-self.y * self.T)* ss.norm.cdf(-self.Vanilla_d1))
        print (self.value)
        return self.value

#Options on Futures
    def Vanilla_European_Call_Forward(self): #Validated
        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
       
        self.F = self.jsondata["Forward"]

        self.Vanilla_d1 = (np.log(self.F/self.K) + (0.5 * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))
        self.value = np.exp(-self.r * self.T)*(self.F * ss.norm.cdf(self.Vanilla_d1) - self.K * ss.norm.cdf(self.Vanilla_d2))
        print (self.value)
        return self.value

    def Vanilla_European_Put_Forward(self): #Validated
        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
       

        self.F = self.jsondata["Forward"]

        self.Vanilla_d1 = (np.log(self.F/self.K) + (0.5 * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))
        self.value = np.exp(-self.r * self.T)*(self.K * ss.norm.cdf(-self.Vanilla_d2) - self.F * ss.norm.cdf(-self.Vanilla_d1) )
        print (self.value)
        return self.value

    def Vanilla_European_Call_Forward_No_Margin(self): #Validated
        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
        
        self.F = self.jsondata["Forward"]
       

        self.Vanilla_d1 = (np.log(self.F/self.K) + (0.5 * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))
        self.value = (self.F * ss.norm.cdf(self.Vanilla_d1) - self.K * ss.norm.cdf(self.Vanilla_d2))
        print (self.value)
        return self.value

    def Vanilla_European_Put_Future_No_Margin(self): #Validated
        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
        self.F = self.jsondata["Forward"]
        self.Vanilla_d1 = (np.log(self.F/self.K) + (0.5 * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))
        self.value = (self.K * ss.norm.cdf(-self.Vanilla_d2) - self.F * ss.norm.cdf(-self.Vanilla_d1))
        print (self.value)
        return self.value



#Binary Options
    def Binary_Cash_In(self): #Validated
        #Payoff is $1
        #S0, K, r, sigma, T, y

         

         
          

        self.K = self.jsondata["Single_Strike"]

        self.Vanilla_d = ((np.log(self.S0/self.K) + (self.b - self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.value = np.exp(-self.r * self.T)* ss.norm.cdf(self.Vanilla_d)

        print (self.value)
        return self.value




    def Binary_Cash_Out(self): #Validated
        #Payoff is $1
        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
         

         
          

        self.Vanilla_d = ((np.log(self.S0/self.K) + (self.b - self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.value = (np.exp(-self.r * self.T)* ss.norm.cdf(-self.Vanilla_d))

        print (self.value)
        return self.value


    def Binary_Asset_In(self): #Validated
        #Payoff is $1
        #S0, K, r, sigma, T, y

        #Universal Params
         

        #Market Params
         
          

        #Contract Specific Numerical Params
        self.K = self.jsondata["Single_Strike"]

        self.Vanilla_d1 = ((np.log(self.S0/self.K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.value = self.S0 * np.exp(-self.y * self.T)* ss.norm.cdf(self.Vanilla_d1)

        print (self.value)
        return self.value

    def Binary_Asset_Out(self):  #Validated
        #Payoff is $1
        #S0, K, r, sigma, T, y

        #Universal Params
         

        #Market Params
         
          

        #Contract Specific Numerical Params
        self.K = self.jsondata["Single_Strike"]

        self.Vanilla_d1 = ((np.log(self.S0/self.K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.value = self.S0 * np.exp(-self.y * self.T)* ss.norm.cdf(-self.Vanilla_d1)

        print (self.value)
        return self.value

    def Binary_Range_Cash_In(self):   #Validated
        #Payoff is $1
        #S0, K, r, sigma, T, y
        self.Upper_K = self.jsondata["Multiple_Strike_Upper"]
        self.Lower_K = self.jsondata["Multiple_Strike_Lower"]
         

         
          

        #Price Cash In Binary at Lower Strike

        self.Lower_d1 = ((np.log(self.S0/self.Lower_K) + (self.b - self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Lower_value = np.exp(-self.r * self.T)* ss.norm.cdf(self.Lower_d1)

        self.Upper_d1 = ((np.log(self.S0/self.Upper_K) + (self.b - self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Upper_value = 1- (np.exp(-self.r * self.T)* ss.norm.cdf(self.Upper_d1))

        self.value = (self.Lower_value + self.Upper_value) -1
        print (self.value)
        return self.value

    def Binary_Range_Cash_Out(self):    #Validated
        #Payoff is $1
        #S0, K, r, sigma, T, y
        self.Upper_K = self.jsondata["Multiple_Strike_Upper"]
        self.Lower_K = self.jsondata["Multiple_Strike_Lower"]
         

         
          

        #Price Cash In Binary at Lower Strike
        self.Lower_d1 = ((np.log(self.S0/self.Lower_K) + (self.b - self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Lower_value = (np.exp(-self.r * self.T)* ss.norm.cdf(-self.Lower_d1))

        self.Upper_d1 = ((np.log(self.S0/self.Upper_K) + (self.b - self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Upper_value = np.exp(-self.r * self.T)* ss.norm.cdf(-self.Upper_d1)

        self.value =  (self.Lower_value - self.Upper_value) + 1
        print (self.value)
        return self.value

    def Supershare_Option(self): #Validated (176)
        """ Prices Supershare option with multiple upper and lower strikes.

        d2 uses the frank method instead of the jeo method of the distribution because ...


        >>> start_time = 5
        >>> jsondata = {Multiple_Strike_Upper: 50, }
        >>> ClosedFormPricing.init(self, jsondata)
        >>> self.jsondata["Multiple_Strike_Upper"] = 50
        >>> price = Supershare_Option(self)
        >>> print type(price)
        float
        """
        #Payoff is S_T/Lower_K
        #S0, r, sigma, T, y
        self.Upper_K = self.jsondata["Multiple_Strike_Upper"]
        self.Lower_K = self.jsondata["Multiple_Strike_Lower"]
         

         
          

        self.Vanilla_d1 = ((np.log(self.S0/self.Lower_K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_d2 = ((np.log(self.S0/self.Upper_K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.value = ((self.S0 *np.exp((self.b - self.r) * self.T))/(self.Lower_K))* (ss.norm.cdf(self.Vanilla_d1) - ss.norm.cdf(self.Vanilla_d2))
        print  self.value
        return self.value

#Touch Options
    def One_Touch_Cash_At_Hit(self): #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])

        self.l = float(self.jsondata["Single_Touch"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
        #Up and in
        if self.S0 < self.l :
            self.eta = -1


        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        #self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.pdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * self.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * self.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        #self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        self.value = self.a5

        print self.value
        return self.value

    def One_Touch_Asset_At_Hit(self): #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])

        self.l = float(self.jsondata["Single_Touch"])
        self.R = self.l
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
        #Up and in
        if self.S0 < self.l :
            self.eta = -1


        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        #self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * self.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * self.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        #self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        self.value = self.a5

        print self.value
        return self.value

    def One_Touch_Cash_At_Expiration(self): #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])

        self.l = float(self.jsondata["Single_Touch"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = -1
        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = 1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        self.value = self.b2 + self.b4

        print self.value
        return self.value

    def One_Touch_Asset_At_Expiration(self):  #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])

        self.l = float(self.jsondata["Single_Touch"])
        self.R = self.l
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = -1
        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = 1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        #self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        #self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        self.value = self.a2 + self.a4

        print self.value
        return self.value

    def No_Touch_Cash(self):    #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])
         

        self.l = float(self.jsondata["Single_Touch"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = 1
        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = -1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        self.value = self.b2 - self.b4

        print self.value
        return self.value

    def No_Touch_Asset(self):    #Validated

        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])

        self.l = float(self.jsondata["Single_Touch"])
        self.R = self.l
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = 1
        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = -1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        #self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        #self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        self.value = self.a2 - self.a4

        print self.value
        return self.value

# def Double_One_Touch(self):

    def Double_No_Touch(self):       #Validated
        #S0, K, r, sigma, T, y, KO, R   #Hui 1996
         

        self.U = float(self.jsondata["Double_Touch_Upper"])
        self.L = float(self.jsondata["Double_Touch_Lower"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        self.Alfa = -0.5 * (2 * self.b / self.sigma**2 - 1)
        self.Beta = -0.25 * (2 * self.b / self.sigma**2 - 1)** 2 - 2 * self.r / self.sigma**2

        self.Z = np.log(self.U / self.L)
        self.sum = 0
        for i in range(1,50):
            self.sum = self.sum + 2 * np.pi * i * self.R / self.Z**2 * (((self.S0 / self.L)**self.Alfa - (-1)**i * (self.S0 / self.U)**self.Alfa) / (self.Alfa**2 + (i * np.pi / self.Z)**2)) * mth.sin(i * np.pi / self.Z * np.log(self.S0 / self.L)) * np.exp(-0.5 * ((i * np.pi / self.Z)**2 - self.Beta) * self.sigma**2 * self.T)

        self.value = self.sum
        print self.value
        return self.value

    def Double_One_Touch(self):   #Validated
        #S0, K, r, sigma, T, y, KO, R   #Hui 1996
         

        self.U = float(self.jsondata["Double_Touch_Upper"])
        self.L = float(self.jsondata["Double_Touch_Lower"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        self.Alfa = -0.5 * (2 * self.b / self.sigma**2 - 1)
        self.Beta = -0.25 * (2 * self.b / self.sigma**2 - 1)** 2 - 2 * self.r / self.sigma**2

        self.Z = np.log(self.U / self.L)
        self.sum = 0
        for i in range(1,50):
            self.sum = self.sum + 2 * np.pi * i * self.R / self.Z**2 * (((self.S0 / self.L)**self.Alfa - (-1)**i * (self.S0 / self.U)**self.Alfa) / (self.Alfa**2 + (i * np.pi / self.Z)**2)) * mth.sin(i * np.pi / self.Z * np.log(self.S0 / self.L)) * np.exp(-0.5 * ((i * np.pi / self.Z)**2 - self.Beta) * self.sigma**2 * self.T)

        self.value = self.R * np.exp(-self.r * self.T) -self.sum
        print self.value
        return self.value

    def One_Touch_Knock_Out(self):   #In Progress (Not Validated)
        #S0, K, r, sigma, T, y, KO, R   #Hui 1996
         

        self.U = float(self.jsondata["Single_Touch"])
        self.L = float(self.jsondata["KO"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        if (self.S0 < self.L and self.S0 < self.U and self.L < self.U) or (self.S0 > self.L and self.S0 > self.U and self.L > self.U):
            self.value = 0
            print "Invalid Knock-Out Placement"

        elif(self.U > self.L):  #if the hit is greater than the KO
            #Switch U and L
            self.upsilon = self.U
            self.U = self.L
            self.L = self.upsilon
            #print(self.U, self.L)
            self.Alfa = -0.5 * (2 * self.b / self.sigma**2 - 1)
            self.Beta = -0.25 * (2 * self.b / self.sigma**2 - 1)** 2 - 2 * self.r / self.sigma**2
            self.Z = np.log(self.U / self.L)
            self.sum = 0
            for i in range(1,50):
                self.sum = self.sum +  2 / (i * np.pi) * ((self.Beta - (i * np.pi / self.Z)**2 * np.exp(-0.5 * ((i * np.pi / self.Z)**2 - self.Beta) * self.sigma**2 * self.T))/ ((i * mth.pi / self.Z)**2 - self.Beta)) * mth.sin(i * np.pi / self.Z * np.log(self.S0 / self.L))
            self.value = self.R * (self.S0 / self.L)**self.Alfa * (self.sum + (1 - np.log(self.S0 / self.L) / self.Z))

        if(self.U < self.L):  #if the hit is less than the KO
            self.Alfa = -0.5 * (2 * self.b / self.sigma**2 - 1)
            self.Beta = -0.25 * (2 * self.b / self.sigma**2 - 1)** 2 - 2 * self.r / self.sigma**2
            self.Z = np.log(self.U / self.L)
            self.sum = 0
            for i in range(1,50):
                self.sum = self.sum +  2 / (i * np.pi) * ((self.Beta - (i * np.pi / self.Z)**2 * np.exp(-0.5 * ((i * np.pi / self.Z)**2 - self.Beta) * self.sigma**2 * self.T))/ ((i * mth.pi / self.Z)**2 - self.Beta)) * mth.sin(i * np.pi / self.Z * np.log(self.S0 / self.L))
            self.value = self.R * (self.S0 / self.L)**self.Alfa * (self.sum + (1 - np.log(self.S0 / self.L) / self.Z))

        print self.value
        return self.value


#Binary Barriers
    def Binary_Cash_In_KI(self):  #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])
         

        self.l = float(self.jsondata["KI"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = 1

        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = 1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        if self.S0 > self.l:
            if self.K > self.l:
                self.value = self.b3
            if self.K < self.l:
                self.value = self.b1- self.b2 +self.b4
        if self.S0 < self.l:
            if self.K > self.l:
                self.value = self.b1
            if self.K < self.l:
                self.value = self.b2 - self.b3 + self.b4

        print self.value
        return self.value

    def Binary_Asset_In_KI(self): #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])
         

        self.l = float(self.jsondata["KI"])
        self.R = self.l
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = 1

        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = 1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        #self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        #self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        if self.S0 > self.l:
            if self.K > self.l:
                self.value = self.a3
            if self.K < self.l:
                self.value = self.a1- self.a2 +self.a4
        if self.S0 < self.l:
            if self.K > self.l:
                self.value = self.a1
            if self.K < self.l:
                self.value = self.a2 - self.a3 + self.a4

        print self.value
        return self.value

    def Binary_Cash_Out_KI(self):   #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])
         

        self.l = float(self.jsondata["KI"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = -1

        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = -1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        if self.S0 > self.l:
            if self.K > self.l:
                self.value = self.b2- self.b3 +self.b4
            if self.K < self.l:
                self.value = self.b1
        if self.S0 < self.l:
            if self.K > self.l:
                self.value = self.b1- self.b2 + self.b4
            if self.K < self.l:
                self.value = self.b3

        print self.value
        return self.value

    def Binary_Asset_Out_KI(self):  #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])
         

        self.l = float(self.jsondata["KI"])
        self.R = self.l
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = -1

        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = -1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        #self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        #self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        if self.S0 > self.l:
            if self.K > self.l:
                self.value = self.a2 - self.a3 + self.a4
            if self.K < self.l:
                self.value = self.a1
        if self.S0 < self.l:
            if self.K > self.l:
                self.value = self.a1 - self.a2 + self.a3
            if self.K < self.l:
                self.value = self.a3

        print self.value
        return self.value

    def Binary_Cash_In_KO(self):     #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])

        self.l = float(self.jsondata["KO"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = 1

        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = 1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        if self.S0 > self.l:
            if self.K > self.l:
                self.value = self.b1 - self.b3
            if self.K < self.l:
                self.value = self.b2 - self.b4
        if self.S0 < self.l:
            if self.K > self.l:
                self.value = 0
            if self.K < self.l:
                self.value = self.b1 - self.b2 + self.b3 - self.b4

        print self.value
        return self.value

    def Binary_Asset_In_KO(self):     #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])
         

        self.l = float(self.jsondata["KO"])
        self.R = self.l
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = 1

        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = 1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        #self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        #self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        if self.S0 > self.l:
            if self.K > self.l:
                self.value = self.a1 - self.a3
            if self.K < self.l:
                self.value = self.a2 - self.a4
        if self.S0 < self.l:
            if self.K > self.l:
                self.value = 0
            if self.K < self.l:
                self.value = self.a1 - self.a2 + self.a3 - self.a4

        print self.value
        return self.value

    def Binary_Asset_Out_KO(self):      #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])
         

        self.l = float(self.jsondata["KO"])
        self.R = self.l
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = -1

        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = -1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        #self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        #self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        #self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        #self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        if self.S0 > self.l:
            if self.K > self.l:
                self.value =  self.a1 - self.a2 + self.a3 - self.a4
            if self.K < self.l:
                self.value = 0
        if self.S0 < self.l:
            if self.K > self.l:
                self.value = self.a2 - self.a4
            if self.K < self.l:
                self.value = self.a1 - self.a3

        print self.value
        return self.value

    def Binary_Cash_Out_KO(self):    #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])
         

        self.l = float(self.jsondata["KO"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        #Down and in
        if self.S0 > self.l :
            self.eta = 1
            self.phi = -1

        #Up and in
        if self.S0 < self.l :
            self.eta = -1
            self.phi = -1

        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2
        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)
        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)

        #self.a1 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1)
        self.b1 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a2 = self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2)
        self.b2 = self.R * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        #self.a3 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1)
        self.b3 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a4 = self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2)
        self.b4 = self.R * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        #self.a5 = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        if self.S0 > self.l:
            if self.K > self.l:
                self.value = self.b1 - self.b2 + self.b3 - self.b4
            if self.K < self.l:
                self.value = 0
        if self.S0 < self.l:
            if self.K > self.l:
                self.value = self.b2 - self.b4
            if self.K < self.l:
                self.value = self.b1 - self.b3

        print self.value
        return self.value



#Knock Out and Knock In Vanilla Options (Reiner-Rubinstein)
    def Vanilla_European_KO_Call(self): #Validated

        #S0, K, r, sigma, T, y, KO, R
        self.K = float(self.jsondata["Single_Strike"])
         

        self.l = float(self.jsondata["KO"])
        self.R = float(self.jsondata["Rebate"])
          
         
          

        #Up and out call
        if self.S0 < self.l :
            self.phi = 1
            self.eta = -1

        #Down and out call
        if self.S0 > self.l :
            self.phi = 1
            self.eta = 1
            print ("true")


        self.mu = (self.b - self.sigma**2 /2)/(self.sigma**2)
        #mu = (b - v ^ 2 / 2) / v ^ 2

        self.lmda = np.sqrt(self.mu**2+ 2*self.r / self.sigma**2)
        #lambda = Sqr(mu ^ 2 + 2 * r / v ^ 2)


        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)


        self.A = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1) - self.phi * self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.B = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2) - self.phi * self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        self.C = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1) - self.phi * self.K * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.D = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2) - self.phi * self.K * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        self.E = self.R * np.exp(-self.r * self.T) * (ss.norm.cdf(self.eta * self.x2 - self.eta * self.sigma * np.sqrt(self.T)) - (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T)))
        self.F = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        """
        #First order derivatives
        self.dA = self.phi* np.exp(-self.y *self.T) * ss.norm.cdf(self.phi*self.x1)
        self.dB = self.phi* np.exp(-self.y *self.T) * ss.norm.cdf(self.phi*self.x2) + np.exp(-self.y + self.T) * ((self.eta*self.x2)/(self.sigma*np.sqrt(self.T))) * (1- (self.K/self.l))
        self.dC = (1/self.S0)* self.phi* 2 * self.mu * (((self.l/self.K)**(2*self.mu))*(self.S0 * np.exp(-self.y *self.T)*((self.l**2)/(self.S0**2))*ss.norm.cdf(self.eta * self.y1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf((self.eta * self.y1) - (self.eta* self.sigma*np.sqrt(self.T))) - self.phi * ((self.l/self.K)**(2*(self.mu + 1))) * np.exp(-self.b * self.T) * ss.norm.cdf(self.eta*self.y1)))
        self.dD = -2*self.mu * (self.phi/self.S0) * (((self.l/self.K)**(2*self.mu)) * (self.S0 * np.exp(-self.y *self.T)*((self.l**2)/(self.S0**2))*ss.norm.cdf(self.eta * self.y2) -  self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.eta * self.y2) * ss.norm.cdf((self.eta * self.y2)  - (self.eta * self.sigma * np.sqrt(self.T)) - self.phi*((self.l/self.K)**(2*self.mu)) *np.exp(-self.y *self.T)) * ss.norm.cdf(self.eta * self.y2 ) - self.phi*self.eta * ((self.l/self.K)**(2*self.mu))* np.exp(-self.y *self.T))*((ss.norm.cdf(self.y2))/(self.sigma*np.sqrt(self.T)))*(1 - (self.K/self.l)))
        self.dE = 2 * (self.R/self.S0) * np.exp(-self.r * self.T) *((self.l/self.S0)**(2*self.mu)) * (ss.norm.cdf(ss.norm.cdf((self.eta * self.y2) - (self.eta* self.sigma*np.sqrt(self.T))))*self.mu +  (self.eta * (ss.norm.cdf((self.eta* self.y2) - (self.T*np.sqrt(self.T)))/(self.T*np.sqrt(self.T)))))
        self.dF = (self.R/self.S0) * ((self.l/self.S0)**(self.mu + self.lmda)) * ((self.mu + self.lmda) * ss.norm.cdf(self.eta *self.z) +(self.mu - self.lmda)*((self.l/self.S0)**(2* self.lmda))) -2*self.eta*self.R*((self.l/self.S0)**(self.mu + self.lmda))*(ss.norm.cdf(self.z)/(self.S0 * (self.sigma * np.sqrt(self.T))))
        """

        #Down and out call
        if self.S0 > self.l :
            if self.K > self.l:
                self.value = self.A - self.C + self.F
            if self.K < self.l:
                self.value = self.B + self.F - self.D

        #Up and out call
        if self.S0 < self.l :
            if self.K > self.l:
                self.value = self.F
            if self.K < self.l:
                self.value = self.A - self.B + self.C - self.D + self.F

        print self.value
        return self.value

    def Vanilla_European_KO_Put(self): #Validated

        #S0, K, r, sigma, T, y, KO, R
        self.K = self.jsondata["Single_Strike"]
            
          
           
        self.l = self.jsondata["KO"]  #Change for other options
        self.R = self.jsondata["Rebate"]  #Change for other options
          

         
          

        self.mu = (self.b - ((self.sigma**2)/2))/(self.sigma**2)
        self.lmda = np.sqrt((self.mu**2)+((2*self.r)/(self.sigma**2)))

        #Down and out put
        if self.S0 > self.l :
            self.phi = -1
            self.eta = 1
            print("1")

        #Up and out put
        if self.S0 < self.l :
            self.phi = -1
            self.eta = -1
            print("2")

        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)


        self.A = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1) - self.phi * self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.B = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2) - self.phi * self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        self.C = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1) - self.phi * self.K * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.D = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2) - self.phi * self.K * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        self.E = self.R * np.exp(-self.r * self.T) * (ss.norm.cdf(self.eta * self.x2 - self.eta * self.sigma * np.sqrt(self.T)) - (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T)))
        self.F = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))



        #Down and out put
        if self.S0 > self.l :
            if self.K > self.l:
                self.value = self.A - self.B + self.C - self.D + self.F
            if self.K < self.l:
                self.value = self.F

        #Up and out put
        if self.S0 < self.l :
            if self.K > self.l:
                self.value = self.B - self.D + self.F
            if self.K < self.l:
                self.value = self.A - self.C + self.F

        print self.value
        return self.value

    def Vanilla_European_KI_Call(self):

        #S0, K, r, sigma, T, y, KO, R
        self.K = self.jsondata["Single_Strike"]
            
          
           
        self.l = self.jsondata["KI"]  #Change for other options
        self.R = self.jsondata["Rebate"]  #Change for other options
          

         
          

        self.mu = (self.b - ((self.sigma**2)/2))/(self.sigma**2)
        self.lmda = np.sqrt((self.mu**2)+((2*self.r)/(self.sigma**2)))

        #Down and In Call
        if self.S0 > self.l :
            self.phi = 1
            self.eta = 1

        #Up and In Call
        if self.S0 < self.l:
            self.phi = 1
            self.eta = -1

        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)


        self.A = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1) - self.phi * self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.B = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2) - self.phi * self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        self.C = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1) - self.phi * self.K * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.D = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2) - self.phi * self.K * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        self.E = self.R * np.exp(-self.r * self.T) * (ss.norm.cdf(self.eta * self.x2 - self.eta * self.sigma * np.sqrt(self.T)) - (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T)))
        self.F = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        if self.K > self.l:
            print("easy")

        #Down and In Call
        if self.S0 > self.l :
            if self.K > self.l:
                self.value = self.C + self.E
            if self.K < self.l:
                self.value = self.A - self.B + self.D + self.E

        #Up and In Call
        if self.S0 < self.l:
            if self.K > self.l:
                self.value = self.A + self.E
            if self.K < self.l:
                self.value = self.B - self.C + self.D + self.E

        print self.value
        return self.value

    def Vanilla_European_KI_Put(self):  #Validated
        #S0, K, r, sigma, T, y, KO, R
        self.K = self.jsondata["Single_Strike"]
            
          
           
        self.l = self.jsondata["KI"]  #Change for other options
        self.R = self.jsondata["Rebate"]  #Change for other options
          

         
          

        self.mu = (self.b - ((self.sigma**2)/2))/(self.sigma**2)
        self.lmda = np.sqrt((self.mu**2)+((2*self.r)/(self.sigma**2)))

        #Down and In Put
        if self.S0 > self.l:
            self.phi = -1
            self.eta = 1

        #Up and In Put
        if self.S0 < self.l:
            self.phi = -1
            self.eta = -1

        self.x1 = np.log(self.S0/self.K)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X1 = Log(S / X) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)
        self.x2 = np.log(self.S0/self.l)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #X2 = Log(S / H) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.y1 = np.log(self.l**2/(self.S0*self.K))/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * (self.sigma * np.sqrt(self.T))
        #y1 = Log(H ^ 2 / (S * X)) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.y2 = np.log(self.l/self.S0)/(self.sigma * np.sqrt(self.T)) + (1 + self.mu) * self.sigma * np.sqrt(self.T)
        #y2 = Log(H / S) / (v * Sqr(T)) + (1 + mu) * v * Sqr(T)

        self.z =  (np.log(self.l/self.S0))/(self.sigma * np.sqrt(self.T)) + self.lmda * self.sigma* np.sqrt(self.T)
        #z = Log(H / S) / (v * Sqr(T)) + lambda * v * Sqr(T)


        self.A = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x1) - self.phi * self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x1 - self.phi * self.sigma * np.sqrt(self.T))
        self.B = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * ss.norm.cdf(self.phi * self.x2) - self.phi * self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.phi * self.x2 - self.phi * self.sigma * np.sqrt(self.T))
        self.C = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y1) - self.phi * self.K * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y1 - self.eta * self.sigma * np.sqrt(self.T))
        self.D = self.phi * self.S0 * np.exp((self.b - self.r) * self.T) * (self.l / self.S0) ** (2 * (self.mu + 1)) * ss.norm.cdf(self.eta * self.y2) - self.phi * self.K * np.exp(-self.r * self.T) * (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T))
        self.E = self.R * np.exp(-self.r * self.T) * (ss.norm.cdf(self.eta * self.x2 - self.eta * self.sigma * np.sqrt(self.T)) - (self.l / self.S0) ** (2 * self.mu) * ss.norm.cdf(self.eta * self.y2 - self.eta * self.sigma * np.sqrt(self.T)))
        self.F = self.R * ((self.l / self.S0) ** (self.mu + self.lmda) * ss.norm.cdf(self.eta * self.z) + (self.l / self.S0) ** (self.mu - self.lmda) * ss.norm.cdf(self.eta * self.z - 2 * self.eta * self.lmda * self.sigma * np.sqrt(self.T)))

        #Down and In Put
        if self.S0 > self.l:
            self.phi = -1
            self.eta = 1
            if self.K > self.l:
                self.value = self.B - self.C + self.D + self.E
            if self.K < self.l:
                self.value = self.A + self.E

        #Up and In Put
        if self.S0 < self.l:
            self.phi = -1
            self.eta = -1
            if self.K > self.l:
                self.value = self.A - self.B + self.D + self.E
            if self.K < self.l:
                self.value = self.C + self.E

        print (self.value)
        return self.value


# Power Options
    def Power_Call(self): #Validated

        self.K = self.jsondata["Single_Strike"]

          
        self.x = self.jsondata["Power_Level"]


        self.pow_d1 = ((np.log(self.S0/(self.K**(1/self.x))) + ((self.r -self.y) + (self.x - 0.5) * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.pow_d2 = self.pow_d1 - (self.x*self.sigma * np.sqrt(self.T))

        self.value = ((self.S0**self.x)* np.exp(((self.x-1)*(self.r+(self.x *((self.sigma**2)/2)))-(self.x*( self.r -  (self.r - self.y))))*self.T)
                        * ss.norm.cdf(self.pow_d1)) -(self.K* np.exp(-self.r*self.T)*ss.norm.cdf(self.pow_d2))

        print (self.value)
        return self.value

    def Power_Put(self): #Validated

        self.K = self.jsondata["Single_Strike"]

          
        self.x = self.jsondata["Power_Level"]

         
          

        self.pow_d1 = ((np.log(self.S0/(self.K**(1/self.x))) + ((self.r -self.q) + (self.x - 0.5) * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.pow_d2 = self.pow_d1 - (self.x*self.sigma * np.sqrt(self.T))

        self.value = (self.K* np.exp(-self.r*self.T)*ss.norm.cdf(-self.pow_d2)) - ((self.S0**self.x)* np.exp(((self.x-1)*(self.r+(self.x *((self.sigma**2)/2)))-(self.x*( self.r -  (self.r - self.q))))*self.T)
                        * ss.norm.cdf(- self.pow_d1))
        print (self.value)
        return self.value

    def Power_Contract(self): #Validated

        self.K = self.jsondata["Single_Strike"]
            
        self.q = self.jsondata["Dividend"]
          
        self.x = self.jsondata["Power_Level"]

         
          

        self.value = ((self.S0/ self.K)**self.x)* np.exp(((((self.r - self.q)- (0.5 * self.sigma**2))*self.x) - (self.r) + ((self.x**2) * (0.5 * self.sigma**2)))* self.T)
        print self.value
        return self.value

    def Capped_Power_Option_Call(self):  #In Progress
        self.K = self.jsondata["Single_Strike"]
            
        self.q = self.jsondata["Dividend"]
          
        self.x = self.jsondata["Power_Level"]
        self.cap = self.jsondata["Cap"]

         
          

        self.pow_e1 = ((np.log(self.S0/(self.K**(1/self.x))) + ((self.r -self.q) + (self.x - 0.5) * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.pow_e2 = self.pow_e1 - (self.x*self.sigma * np.sqrt(self.T))

        self.pow_e3 = ((np.log(self.S0/((self.cap + self.K)**(1/self.x))) + ((self.r -self.q) + (self.x - 0.5) * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.pow_e4 = self.pow_e3 - (self.x*self.sigma * np.sqrt(self.T))

        self.value = (((self.S0**self.x)* np.exp(((self.x-1)*(self.r+(self.x *((self.sigma**2)/2)))-(self.x*( self.r -  (self.r - self.q))))*self.T)
                        * (ss.norm.cdf(self.pow_e1)- ss.norm.cdf(self.pow_e3)))
                     - (np.exp(-self.r*self.T)*((self.K *ss.norm.cdf(self.pow_e2)) - ((self.cap + self.K)*ss.norm.cdf(self.pow_e4)))))

        print (self.value)
        return self.value

    def Capped_Power_Option_Put(self):  #In Progress
        self.K = self.jsondata["Single_Strike"]
            
        self.q = self.jsondata["Dividend"]
          
        self.x = self.jsondata["Power_Level"]
        self.cap = self.jsondata["Cap"]

         
          

        self.pow_e1 = ((np.log(self.S0/(self.K**(1/self.x))) + ((self.r -self.q) + (self.x - 0.5) * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.pow_e2 = self.pow_e1 - (self.x*self.sigma * np.sqrt(self.T))

        self.pow_e3 = ((np.log(self.S0/((self.cap + self.K)**(1/self.x))) + ((self.r -self.q) + (self.x - 0.5) * self.sigma**2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.pow_e4 = self.pow_e3 - (self.x*self.sigma * np.sqrt(self.T))

        self.value = (((self.S0**self.x)* np.exp(((self.x-1)*(self.r+(self.x *((self.sigma**2)/2)))-(self.x*( self.r -  (self.r - self.q))))*self.T)
                        * (ss.norm.cdf(self.pow_e1)- ss.norm.cdf(self.pow_e3)))
                     - (np.exp(-self.r*self.T)*((self.K *ss.norm.cdf(self.pow_e2)) - ((self.cap + self.K)*ss.norm.cdf(self.pow_e4)))))

        print (self.value)
        return self.value


# Asian Options (Kemna and Vorst)
    def Average_Fixed_Strike_Arithmetic_Call(self): #Validated

        #Universal Params
         

        if self.b == 0:
            self.b = 0.001

        #Market Params
         
          

        #Contract Specific Numerical Params
        self.K = self.jsondata["Single_Strike"]
        self.t = self.jsondata["Forward_Start"]


        self.M1 = (np.exp(self.b *self.T) - np.exp(self.b *self.t))/(self.b * (self.T - self.t))

        self.M2 = ((2* np.exp((2 * self.b + self.sigma**2)*self.T))/((self.b + self.sigma**2 )*(2* self.b + self.sigma**2)*((self.T - self.t )**2))) + \
        (((2* np.exp((2 * self.b + self.sigma**2)*self.t))/(self.b * ((self.T - self.t)**2))) * \
         ((1/(2* self.b + self.sigma**2)) - ((np.exp(self.b *(self.T - self.t))) / (self.b + self.sigma**2)) ))

        self.b_A = np.log(self.M1)/self.T
        self.sigma_A = np.sqrt((np.log(self.M2)/self.T) - (2 * self.b_A))

        self.d1 = (np.log(self.S0/self.K) + (self.b_A + (.5*self.sigma_A**2)) * self.T)/(self.sigma_A * np.sqrt(self.T))
        self.d2 = self.d1 - (self.sigma_A * np.sqrt(self.T))

        self.value = (self.S0 * np.exp((self.b_A - self.r)* self.T)* ss.norm.cdf(self.d1)) - (self.K * np.exp(-self.r * self.T) *ss.norm.cdf(self.d2))

        print(self.value)
        return self.value

    def Average_Fixed_Strike_Arithmetic_Put(self): #Validated

        #Universal Params
         

        if self.b == 0:
            self.b = 0.001

        #Market Params
         
          

        #Contract Specific Numerical Params
        self.K = self.jsondata["Single_Strike"]
        self.t = self.jsondata["Forward_Start"]


        self.M1 = (np.exp(self.b *self.T) - np.exp(self.b *self.t))/(self.b * (self.T - self.t))

        self.M2 = ((2* np.exp((2 * self.b + self.sigma**2)*self.T))/((self.b + self.sigma**2 )*(2* self.b + self.sigma**2)*((self.T - self.t )**2))) + \
        (((2* np.exp((2 * self.b + self.sigma**2)*self.t))/(self.b * ((self.T - self.t)**2))) * \
         ((1/(2* self.b + self.sigma**2)) - ((np.exp(self.b *(self.T - self.t))) / (self.b + self.sigma**2)) ))

        self.b_A = np.log(self.M1)/self.T
        self.sigma_A = np.sqrt((np.log(self.M2)/self.T) - (2 * self.b_A))

        self.d1 = (np.log(self.S0/self.K) + (self.b_A + (.5*self.sigma_A**2)) * self.T)/(self.sigma_A * np.sqrt(self.T))
        self.d2 = self.d1 - (self.sigma_A * np.sqrt(self.T))

        self.value = (self.K * np.exp(-self.r * self.T) *ss.norm.cdf(-self.d2)) - (self.S0 * np.exp((self.b_A - self.r)* self.T)* ss.norm.cdf(-self.d1))

        print(self.value)
        return self.value

    def Average_Fixed_Strike_Geometric_Call(self): #Validated
            #Universal Params
                
              
               
              

            #Market Params
             
              

            #Contract Specific Numerical Params
            self.K = self.jsondata["Single_Strike"]

            self.sigma_A = self.sigma / (np.sqrt(3))
            self.b_A = 0.5 * (self.b - ((self.sigma**2)/6))
            self.d1 = (np.log(self.S0/self.K) + (self.b_A + (0.5*self.sigma_A**2)) * self.T)/(self.sigma_A * np.sqrt(self.T))
            self.d2 = self.d1 - (self.sigma_A * np.sqrt(self.T))

            self.value = self.S0 * np.exp((self.b_A - self.r)* self.T)* ss.norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) *ss.norm.cdf(self.d2)
            print (self.value)
            return self.value

    def Average_Fixed_Strike_Geometric_Put(self): #Validated

        #Universal Params
         

        #Market Params
         
          

        #Contract Specific Numerical Params
        self.K = self.jsondata["Single_Strike"]

        self.sigma_A = self.sigma / (np.sqrt(3))
        self.b_A = 0.5 * (self.b - ((self.sigma**2)/6))
        self.d1 = (np.log(self.S0/self.K) + (self.b_A + (0.5*self.sigma_A**2)) * self.T)/(self.sigma_A * np.sqrt(self.T))
        self.d2 = self.d1 - (self.sigma_A * np.sqrt(self.T))

        self.value = self.K * np.exp(-self.r * self.T) *ss.norm.cdf(-self.d2) - self.S0 * np.exp((self.b_A - self.r)* self.T)* ss.norm.cdf(-self.d1)
        print (self.value)
        return self.value

    def Average_Floating_Strike_Arithmetic_Call(self): #Validated
         #Universal Params
        self.r = self.jsondata["Dividend_Rate"]
          
        self.y = self.jsondata["Interest_Rate"]
          

        if self.b == 0:
            self.b = 0.001

        #Market Params
         
          

        #Contract Specific Numerical Params
        self.K = self.S0 #ASIAN SYM
        self.t = self.jsondata["Forward_Start"]


        self.M1 = (np.exp(self.b *self.T) - np.exp(self.b *self.t))/(self.b * (self.T - self.t))

        self.M2 = ((2* np.exp((2 * self.b + self.sigma**2)*self.T))/((self.b + self.sigma**2 )*(2* self.b + self.sigma**2)*((self.T - self.t )**2))) + \
        (((2* np.exp((2 * self.b + self.sigma**2)*self.t))/(self.b * ((self.T - self.t)**2))) * \
         ((1/(2* self.b + self.sigma**2)) - ((np.exp(self.b *(self.T - self.t))) / (self.b + self.sigma**2)) ))

        self.b_A = np.log(self.M1)/self.T
        self.sigma_A = np.sqrt((np.log(self.M2)/self.T) - (2 * self.b_A))

        self.d1 = (np.log(self.S0/self.K) + (self.b_A + (.5*self.sigma_A**2)) * self.T)/(self.sigma_A * np.sqrt(self.T))
        self.d2 = self.d1 - (self.sigma_A * np.sqrt(self.T))

        self.value = (self.K * np.exp(-self.r * self.T) *ss.norm.cdf(-self.d2)) - (self.S0 * np.exp((self.b_A - self.r)* self.T)* ss.norm.cdf(-self.d1))

        print(self.value)
        return self.value

    def Average_Floating_Strike_Arithmetic_Put(self, Single_Strike): #Validated

        #Universal Params
        self.r = self.Dividend_Rate

        self.y = self.Interest_Rate


        if self.b == 0:
            self.b = 0.001

        #Market Params
        self.S0 = Single_Strike


        #Contract Specific Numerical Params
        self.K = self.S0
        self.t = self.jsondata["Forward_Start"]


        self.M1 = (np.exp(self.b *self.T) - np.exp(self.b *self.t))/(self.b * (self.T - self.t))

        self.M2 = ((2* np.exp((2 * self.b + self.sigma**2)*self.T))/((self.b + self.sigma**2 )*(2* self.b + self.sigma**2)*((self.T - self.t )**2))) + \
                  (((2* np.exp((2 * self.b + self.sigma**2)*self.t))/(self.b * ((self.T - self.t)**2))) * \
                   ((1/(2* self.b + self.sigma**2)) - ((np.exp(self.b *(self.T - self.t))) / (self.b + self.sigma**2)) ))

        self.b_A = np.log(self.M1)/self.T
        self.sigma_A = np.sqrt((np.log(self.M2)/self.T) - (2 * self.b_A))

        self.d1 = (np.log(self.S0/self.K) + (self.b_A + (.5*self.sigma_A**2)) * self.T)/(self.sigma_A * np.sqrt(self.T))
        self.d2 = self.d1 - (self.sigma_A * np.sqrt(self.T))

        self.value = (self.S0 * np.exp((self.b_A - self.r)* self.T)* ss.norm.cdf(self.d1)) - (self.K * np.exp(-self.r * self.T) *ss.norm.cdf(self.d2))

        print(self.value)
        return self.value

    def Average_Floating_Strike_Geometric_Call(self): #Validated
        #Universal Params
        self.r = self.jsondata["Dividend_Rate"]
          
        self.y = self.jsondata["Interest_Rate"]
          

        if self.b == 0:
            self.b = 0.001

        #Market Params
         
          

        #Contract Specific Numerical Params
        self.K = self.S0 #ASIAN SYM

        self.sigma_A = self.sigma / (np.sqrt(3))
        self.b_A = 0.5 * (self.b - ((self.sigma**2)/6))
        self.d1 = (np.log(self.S0/self.K) + (self.b_A + (0.5*self.sigma_A**2)) * self.T)/(self.sigma_A * np.sqrt(self.T))
        self.d2 = self.d1 - (self.sigma_A * np.sqrt(self.T))

        self.value = self.K * np.exp(-self.r * self.T) *ss.norm.cdf(-self.d2) - self.S0 * np.exp((self.b_A - self.r)* self.T)* ss.norm.cdf(-self.d1)
        print (self.value)
        return self.value

    def Average_Floating_Strike_Geometric_Put(self): #Validated

        #Universal Params
        self.r = self.jsondata["Dividend_Rate"]
          
        self.y = self.jsondata["Interest_Rate"]
          

        if self.b == 0:
            self.b = 0.001

        #Market Params
        self.S0 = self.jsondata["Single_Strike"]
          

        #Contract Specific Numerical Params
        self.K = self.S0



        self.sigma_A = self.sigma / (np.sqrt(3))
        self.b_A = 0.5 * (self.b - ((self.sigma**2)/6))
        self.d1 = (np.log(self.S0/self.K) + (self.b_A + (0.5*self.sigma_A**2)) * self.T)/(self.sigma_A * np.sqrt(self.T))
        self.d2 = self.d1 - (self.sigma_A * np.sqrt(self.T))

        self.value = self.S0 * np.exp((self.b_A - self.r)* self.T)* ss.norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) *ss.norm.cdf(self.d2)
        print (self.value)
        return self.value


#Other Options

    def Log_Option_Call(self): #Validated
        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
         

         
          

        self.Vanilla_d2 = ((np.log(self.S0/self.K) + (self.b - self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))

        self.value = np.exp(-self.r*self.T)* ss.norm.pdf(self.Vanilla_d2) * self.sigma * np.sqrt(self.T)  + \
                     (np.exp(-self.r*self.T)*(np.log(self.S0/self.K) + (self.b - self.sigma**2 / 2)*self.T) * ss.norm.cdf(self.Vanilla_d2))
        print self.value
        return self.value

    def Forward_Start_Call(self):  #Validated

         

         
          

        self.alpha = self.jsondata["Moneyness"]
        self.t = self.jsondata["Forward_Start"]

        self.Vanilla_d1 = ((np.log(1/self.alpha) + (self.b + self.sigma**2 / 2) * (self.T - self.t))/(self.sigma * np.sqrt(self.T- self.t)))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T-self.t))


        self.value = self.S0 *np.exp((self.b- self.r)*self.t) * (np.exp(((self.b- self.r)*(self.T - self.t)))* ss.norm.cdf(self.Vanilla_d1)
                     - self.alpha * np.exp(-self.r *(self.T - self.t)) * ss.norm.cdf(self.Vanilla_d2))
        print self.value
        return self.value

    def Forward_Start_Put(self): #Validated

         

         
          

        self.alpha = self.jsondata["Moneyness"]
        self.t = self.jsondata["Forward_Start"]

        self.Vanilla_d1 = ((np.log(1/self.alpha) + (self.b + self.sigma**2 / 2) * (self.T - self.t))/(self.sigma * np.sqrt(self.T- self.t)))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T-self.t))


        self.value = self.S0 *np.exp((self.b- self.r)*self.t) * ((self.alpha * np.exp(-self.r *(self.T - self.t)) * ss.norm.cdf(-self.Vanilla_d2)) - (np.exp(((self.b- self.r)*(self.T - self.t)))* ss.norm.cdf(-self.Vanilla_d1)))

        print self.value
        return self.value

    def Variable_Purchase_Call(self):

        self.K = self.jsondata["Single_Strike"]
        self.Upper_K = self.jsondata["Multiple_Strike_Upper"]
        self.Lower_K = self.jsondata["Multiple_Strike_Lower"]
        self.Discount = self.jsondata["Discount"]

         

         
          

        self.N_min = self.K/(self.Upper_K* (1 - self.Discount ))
        self.N_max = self.K/(self.Lower_K* (1 - self.Discount ))

        self.d1 = ((np.log(self.S0/self.Upper_K) + (self.b + self.sigma**2 / 2) * (self.T))/(self.sigma * np.sqrt(self.T)))
        self.d2 = self.d1 - (self.sigma* np.sqrt(self.T))
        self.d3 = ((np.log(self.S0/self.Lower_K) + (self.b + self.sigma**2 / 2) * (self.T))/(self.sigma * np.sqrt(self.T)))
        self.d4 = self.d3 - (self.sigma* np.sqrt(self.T))
        self.d5 = ((np.log(self.S0/(self.Lower_K * (1- self.Discount))) + (self.b + self.sigma**2 / 2) * (self.T))/(self.sigma * np.sqrt(self.T)))
        self.d6 = self.d5 - (self.sigma* np.sqrt(self.T))



        self.value = self.K *self.Discount / (1- self.Discount) * np.exp(-self.r *self.T) + self.N_min * (self.S0 *np.exp((self.b- self.r)*self.T) * ss.norm.cdf(self.d1)  - self.Upper_K * np.exp(-self.r *self.T) * ss.norm.cdf(self.d2)) - self.N_max* (self.Lower_K * np.exp(-self.r * self.T) * ss.norm.cdf(-self.d4) - self.S0 * np.exp((self.b- self.r)*self.T) * ss.norm.cdf(-self.d3)) + self.N_max* (self.Lower_K * (1 - self.Discount) * np.exp(-self.r * self.T) * ss.norm.cdf(-self.d6)- self.S0 * np.exp((self.b- self.r)*self.T) * ss.norm.cdf(-self.d5))

#Chooser Options
    def Simple_Chooser(self):     #Validated
        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
         
        self.t = self.jsondata["Choice"]

         
          

        self.Vanilla_d = ((np.log(self.S0/self.K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_y = (np.log(self.S0/self.K) + (self.b*self.T) + (((self.sigma**2)*self.t)/2)) / (self.sigma * np.sqrt(self.t))

        self.value = (self.S0 *np.exp((self.b - self.r) * self.T)* ss.norm.cdf(self.Vanilla_d)) - (self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.Vanilla_d - self.sigma* np.sqrt(self.T))) - (self.S0* np.exp((self.b - self.r)*self.T)* ss.norm.cdf(-self.Vanilla_y)) + (self.K *np.exp(-self.r * self.T)*ss.norm.cdf(-self.Vanilla_y + self.sigma * np.sqrt(self.t)))
        print self.value
        return self.value
#def Complex_Chooser(self):


#Compound Options (options on options)
    #def Compound_Call_On_Call(self):
    #def Compound_Call_On_Put(self):
    #def Compound_Put_On_Call(self):
    #def Compound_Put_On_Put(self):


#Pay Later
    def Pay_Later_Call(self): #Turnbull and Wakeman (1991)

        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
         

         
          

        self.Vanilla_d1 = (np.log(self.S0/self.K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))
        self.C_T = self.S0 *np.exp(-self.b * self.T) * (ss.norm.cdf(self.Vanilla_d1)/ss.norm.cdf(self.Vanilla_d2)) - self.K
        self.value = (self.S0 *np.exp(-self.y * self.T)* ss.norm.cdf(self.Vanilla_d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.Vanilla_d2)) - (self.C_T * ss.norm.cdf(self.Vanilla_d2))
        print(self.value)
        return self.value

    def Pay_Later_Put(self): #Turnbull and Wakeman (1991)   needs to be converted

        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
         

         
          

        self.Vanilla_d1 = (np.log(self.S0/self.K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))
        self.P_T = self.S0 *np.exp(-self.b * self.T) * (ss.norm.cdf(self.Vanilla_d1)/ss.norm.cdf(self.Vanilla_d2)) - self.K
        self.value = (self.S0 *np.exp(-self.y * self.T)* ss.norm.cdf(self.Vanilla_d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.Vanilla_d2)) - (self.P_T * ss.norm.cdf(self.Vanilla_d2))
        print(self.value)
        return self.value


#Lookbacks
    def Floating_Strike_Lookback_Call(self): #Validated
        #S0, K, r, sigma, T, y
         

        if self.b == 0:
            self.b = 0.00001

         
          

        #previous max price
        self.S_Max = self.jsondata["S_Max"]
        #previous min price
        self.S_Min = self.jsondata["S_Min"]

        self.Vanilla_a1 = ((np.log(self.S0/self.S_Min) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_a2 = self.Vanilla_a1 - (self.sigma* np.sqrt(self.T))

        self.value = ((self.S0 *np.exp((self.b - self.r) * self.T)* ss.norm.cdf(self.Vanilla_a1)) - (self.S_Min * np.exp(-self.r * self.T) * ss.norm.cdf(self.Vanilla_a2))) + ((self.S0 * np.exp(-self.r * self.T) * ((self.sigma**2)/(2*self.b))) * ((((self.S0/self.S_Min)**((-2*self.b)/(self.sigma**2))) * (ss.norm.cdf( - self.Vanilla_a1 + (((2*self.b)/self.sigma) * np.sqrt(self.T))))) - (np.exp(self.b * self.T) * ss.norm.cdf( - self.Vanilla_a1))))

        print(self.value)
        return(self.value)

    def Floating_Strike_Lookback_Put(self): #Validated
                #S0, K, r, sigma, T, y
         

        if self.b == 0:
            self.b = 0.00001

         
          

        #previous max price
        self.S_Max = self.jsondata["S_Max"]
        #previous min price
        self.S_Min = self.jsondata["S_Min"]

        self.Vanilla_b1 = ((np.log(self.S0/self.S_Max) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_b2 = self.Vanilla_b1 - (self.sigma* np.sqrt(self.T))

        self.value = ((self.S_Max * np.exp(-self.r * self.T) * ss.norm.cdf(-self.Vanilla_b2))) - (self.S0 *np.exp((self.b - self.r) * self.T)* ss.norm.cdf(-self.Vanilla_b1)) + ((self.S0 * np.exp(-self.r * self.T) * ((self.sigma**2)/(2*self.b))) * ((-((self.S0/self.S_Max)**((-2*self.b)/(self.sigma**2))) * (ss.norm.cdf( self.Vanilla_b1 + (((-2*self.b)/self.sigma) * np.sqrt(self.T))))) + (np.exp(self.b * self.T) * ss.norm.cdf(self.Vanilla_b1))))

        print(self.value)
        return(self.value)

    def Fixed_Strike_Lookback_Call(self, Single_Strike, S_Max, S_Min): #Validated

        #S0, K, r, sigma, T, y
         
          
            
          
           


          

        if self.b == 0:
            self.b = 0.00001

        self.K = self.jsondata["Single_Strike"]

        #previous max price
        self.S_Max = self.jsondata["S_Max"]
        #previous min price
        self.S_Min = self.jsondata["S_Min"]

        self.Vanilla_d1 = ((np.log(self.S0/self.K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))

        self.Vanilla_e1 = ((np.log(self.S0/self.S_Max) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_e2 = self.Vanilla_e1 - (self.sigma* np.sqrt(self.T))

        if(self.K <= self.S_Max):
            self.value = (np.exp(-self.r * self.T)*(self.S_Max - self.K)) +  (self.S0 *np.exp((self.b -self.r) * self.T) *ss.norm.cdf(self.Vanilla_e1)) - (self.S_Max* np.exp(-self.r*self.T) * ss.norm.cdf(self.Vanilla_e2)) +(self.S0 * np.exp(-self.r * self.T) * ((self.sigma**2)/(2*self.b))) * ((-((self.S0/self.S_Max)**((-2*self.b)/(self.sigma**2))) * (ss.norm.cdf( self.Vanilla_e1 - (((2*self.b)/self.sigma) * np.sqrt(self.T))))) +(np.exp(self.b * self.T) * ss.norm.cdf( self.Vanilla_e1)))
        else:
            self.value = (self.S0 *np.exp((-self.b - self.r) * self.T)* ss.norm.cdf(self.Vanilla_d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(self.Vanilla_d2))+ ((self.S0 * np.exp(-self.r * self.T) * ((self.sigma**2)/(2*self.b))) * ((-((self.S0/self.K)**((-2*self.b)/(self.sigma**2))) * (ss.norm.cdf( self.Vanilla_d1 - (((2*self.b)/self.sigma) * np.sqrt(self.T))))) + (np.exp(self.b * self.T) * ss.norm.cdf( self.Vanilla_d1))))

        print self.value
        return self.value

    def Fixed_Strike_Lookback_Put(self): #Validated
        #S0, K, r, sigma, T, y
        self.K = self.jsondata["Single_Strike"]
         

        if self.b == 0:
            self.b = 0.00001

         
          

        #previous min price
        self.S_Min = self.jsondata["S_Min"]

        self.Vanilla_d1 = ((np.log(self.S0/self.K) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_d2 = self.Vanilla_d1 - (self.sigma* np.sqrt(self.T))

        self.Vanilla_f1 = ((np.log(self.S0/self.S_Min) + (self.b + self.sigma**2 / 2) * self.T)/(self.sigma * np.sqrt(self.T)))
        self.Vanilla_f2 = self.Vanilla_f1 - (self.sigma* np.sqrt(self.T))

        if(self.K >= self.S_Min):
            self.value = (np.exp(-self.r * self.T)*(self.K - self.S_Min )) -  (self.S0 *np.exp((self.b -self.r) * self.T) *ss.norm.cdf(-self.Vanilla_f1)) + (self.S_Min* np.exp(-self.r*self.T) * ss.norm.cdf(-self.Vanilla_f2)) +(self.S0 * np.exp(-self.r * self.T) * ((self.sigma**2)/(2*self.b))) * ((((self.S0/self.S_Min)**((-2*self.b)/(self.sigma**2))) * (ss.norm.cdf( -self.Vanilla_f1 + (((2*self.b)/self.sigma) * np.sqrt(self.T))))) -(np.exp(self.b * self.T) * ss.norm.cdf(-self.Vanilla_f1)))
        else:
            self.value = ((self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-self.Vanilla_d2)) -
                     (self.S0 *np.exp((self.b - self.r )* self.T)* ss.norm.cdf(-self.Vanilla_d1)))\
                      + ((self.S0 * np.exp(-self.r * self.T) * ((self.sigma**2)/(2*self.b))) * \
                    ((((self.S0/self.K)**((-2*self.b)/(self.sigma**2))) *
                      (ss.norm.cdf( -self.Vanilla_d1 + (((2*self.b)/self.sigma) * np.sqrt(self.T)))))- \
                    (np.exp(self.b * self.T) * ss.norm.cdf(-self.Vanilla_d1))))

        print self.value
        return self.value

    def Extreme_Spread_Call(self): #Validated

        #S0, K, r, sigma, T, y
         

        if self.b == 0:
            self.b = 0.00001

         
          

        #previous max price
        self.S_Max = self.jsondata["S_Max"]
        #previous min price
        self.S_Min = self.jsondata["S_Min"]
        self.t = self.jsondata["Forward_Start"]

        #detemined by option type
        self.eta = 1
        self.phi = 1
        self.M0 = self.S_Max

        self.m = np.log(self.M0/self.S0)
        self.mu1 = self.b-(0.5* (self.sigma**2))
        self.mu2 = self.b+(0.5* (self.sigma**2))

        self.value = self.eta *((self.S0 *np.exp((self.b - self.r )* self.T) * (1+((self.sigma**2)/(2*self.b))) * (ss.norm.cdf(self.eta* ((-self.m + self.mu2*self.T )/(self.sigma* np.sqrt(self.T)))))) -

                    (np.exp(-self.b* (self.T - self.t)) * self.S0 * np.exp((self.b- self.r)*self.T) *  ((1+((self.sigma**2)/(2*self.b)))* (ss.norm.cdf(self.eta* ((-self.m + self.mu2*self.t )/(self.sigma* np.sqrt(self.t)))))))

                    + (np.exp(-self.r* self.T)* self.M0 * ss.norm.cdf(self.eta* ((self.m - self.mu1*self.T )/(self.sigma* np.sqrt(self.T)))))

                    - (np.exp(-self.r* self.T)* self.M0 * ((self.sigma**2)/(2*self.b)) * np.exp((2*self.mu1*self.m)/(self.sigma**2)) * (ss.norm.cdf(self.eta* ((-self.m - self.mu1*self.T )/(self.sigma* np.sqrt(self.T))))))

                    - (np.exp(-self.r* self.T)* self.M0 * (ss.norm.cdf(self.eta* ((self.m - self.mu1*self.t )/(self.sigma* np.sqrt(self.t))))))

                    + (np.exp(-self.r* self.T)* self.M0 * ((self.sigma**2)/(2*self.b)) * np.exp((2*self.mu1*self.m)/(self.sigma**2)) * (ss.norm.cdf(self.eta* ((-self.m - self.mu1*self.t )/(self.sigma* np.sqrt(self.t)))))))

    def Extreme_Spread_Put(self): #Validated
            #S0, K, r, sigma, T, y
                
              
               
              

            if self.b == 0:
                self.b = 0.00001

             
              

            #previous max price
            self.S_Max = self.jsondata["S_Max"]
            #previous min price
            self.S_Min = self.jsondata["S_Min"]
            self.t = self.jsondata["Forward_Start"]

            #detemined by option type
            self.eta = -1
            self.phi = 1
            self.M0 = self.S_Min

            self.m = np.log(self.M0/self.S0)
            self.mu1 = self.b-(0.5* (self.sigma**2))
            self.mu2 = self.b+(0.5* (self.sigma**2))

            self.value = self.eta *((self.S0 *np.exp((self.b - self.r )* self.T) * (1+((self.sigma**2)/(2*self.b))) * (ss.norm.cdf(self.eta* ((-self.m + self.mu2*self.T )/(self.sigma* np.sqrt(self.T)))))) -

                        (np.exp(-self.b* (self.T - self.t)) * self.S0 * np.exp((self.b- self.r)*self.T) *  ((1+((self.sigma**2)/(2*self.b)))* (ss.norm.cdf(self.eta* ((-self.m + self.mu2*self.t )/(self.sigma* np.sqrt(self.t)))))))

                        + (np.exp(-self.r* self.T)* self.M0 * ss.norm.cdf(self.eta* ((self.m - self.mu1*self.T )/(self.sigma* np.sqrt(self.T)))))

                        - (np.exp(-self.r* self.T)* self.M0 * ((self.sigma**2)/(2*self.b)) * np.exp((2*self.mu1*self.m)/(self.sigma**2)) * (ss.norm.cdf(self.eta* ((-self.m - self.mu1*self.T )/(self.sigma* np.sqrt(self.T))))))

                        - (np.exp(-self.r* self.T)* self.M0 * (ss.norm.cdf(self.eta* ((self.m - self.mu1*self.t )/(self.sigma* np.sqrt(self.t))))))

                        + (np.exp(-self.r* self.T)* self.M0 * ((self.sigma**2)/(2*self.b)) * np.exp((2*self.mu1*self.m)/(self.sigma**2)) * (ss.norm.cdf(self.eta* ((-self.m - self.mu1*self.t )/(self.sigma* np.sqrt(self.t)))))))

            print self.value
            return self.value

    def Reverse_Extreme_Spread_Call(self): #Validated
    #S0, K, r, sigma, T, y

                
              
               
              

            if self.b == 0:
                self.b = 0.00001

             
              

            #previous max price
            self.S_Max = self.jsondata["S_Max"]
            #previous min price
            self.S_Min = self.jsondata["S_Min"]
            self.t = self.jsondata["Forward_Start"]

            #detemined by option type
            self.eta = 1
            self.phi = -1
            self.M0 = self.S_Min

            self.m = np.log(self.M0/self.S0)
            self.mu1 = self.b-(0.5* (self.sigma**2))
            self.mu2 = self.b+(0.5* (self.sigma**2))

            self.value = -self.eta *((self.S0 *np.exp((self.b - self.r )* self.T) * (1+((self.sigma**2)/(2*self.b))) * (ss.norm.cdf(self.eta* ((self.m - self.mu2*self.T )/(self.sigma* np.sqrt(self.T))))))

                        + (np.exp(-self.r* self.T) * self.M0 * (ss.norm.cdf(self.eta* ((-self.m + self.mu1*self.T )/(self.sigma* np.sqrt(self.T))))))

                        - (np.exp(-self.r* self.T)* self.M0 * (((self.sigma**2)/(2*self.b)) *  np.exp((2*self.mu1*self.m)/(self.sigma**2))) * (ss.norm.cdf(self.eta* ((self.m + self.mu1*self.T)/(self.sigma* np.sqrt(self.T))))))

                        - (self.S0* np.exp((self.b -self.r)* self.T)* (1+((self.sigma**2)/(2*self.b))) * (ss.norm.cdf(self.eta* ((-self.mu2 * (self.T-self.t))/(self.sigma* np.sqrt(self.T-self.t))))))

                        - (np.exp(-self.b*(self.T- self.t))* (self.S0* np.exp((self.b -self.r)* self.T)) * (1+((self.sigma**2)/(2*self.b))) * (ss.norm.cdf(self.eta* ((self.mu1 * (self.T-self.t))/(self.sigma* np.sqrt(self.T-self.t)))))))

            print self.value
            return self.value

    def Reverse_Extreme_Spread_Put(self):
    #S0, K, r, sigma, T, y
                
              
               
              

            if self.b == 0:
                self.b = 0.00001

             
              

            #previous max price
            self.S_Max = self.jsondata["S_Max"]
            #previous min price
            self.S_Min = self.jsondata["S_Min"]
            self.t = self.jsondata["Forward_Start"]

            #detemined by option type
            self.eta = -1
            self.phi = -1
            self.M0 = self.S_Max

            self.m = np.log(self.M0/self.S0)
            self.mu1 = self.b-(0.5* (self.sigma**2))
            self.mu2 = self.b+(0.5* (self.sigma**2))

            self.value = -self.eta *((self.S0 *np.exp((self.b - self.r )* self.T) * (1+((self.sigma**2)/(2*self.b))) * (ss.norm.cdf(self.eta* ((self.m - self.mu2*self.T )/(self.sigma* np.sqrt(self.T))))))

                        + (np.exp(-self.r* self.T) * self.M0 * (ss.norm.cdf(self.eta* ((-self.m + self.mu1*self.T )/(self.sigma* np.sqrt(self.T))))))

                        - (np.exp(-self.r* self.T)* self.M0 * (((self.sigma**2)/(2*self.b)) *  np.exp((2*self.mu1*self.m)/(self.sigma**2))) * (ss.norm.cdf(self.eta* ((self.m + self.mu1*self.T)/(self.sigma* np.sqrt(self.T))))))

                        - (self.S0* np.exp((self.b -self.r)* self.T)* (1+((self.sigma**2)/(2*self.b))) * (ss.norm.cdf(self.eta* ((-self.mu2 * (self.T-self.t))/(self.sigma* np.sqrt(self.T-self.t))))))

                        - (np.exp(-self.b*(self.T- self.t))* (self.S0* np.exp((self.b -self.r)* self.T)) * (1+((self.sigma**2)/(2*self.b))) * (ss.norm.cdf(self.eta* ((self.mu1 * (self.T-self.t))/(self.sigma* np.sqrt(self.T-self.t)))))))

            print self.value
            return self.value


#Accruals

    def Accrual_In (self):
         

         
          

        self.K = self.jsondata["Single_Accrual"]

        self.t_step = np.linspace(0.00274,self.T, 365)
        #print self.t_step
        self.Vanilla_d = ((np.log(self.S0/self.K) + (self.b - self.sigma**2 / 2) * self.t_step)/(self.sigma * np.sqrt(self.t_step)))
        self.x = np.exp(-self.r * self.t_step)* ss.norm.cdf(self.Vanilla_d)
        #print self.x
        self.value = np.mean(self.x)

        print (self.value)
        return self.value

    def Accrual_Out (self):
         

         
          

        self.K = self.jsondata["Single_Accrual"]

        self.t_step = np.linspace(0.00274,self.T, 365)
        #print self.t_step
        self.Vanilla_d = ((np.log(self.S0/self.K) + (self.b - self.sigma**2 / 2) * self.t_step)/(self.sigma * np.sqrt(self.t_step)))
        self.x = (np.exp(-self.r * self.t_step)* ss.norm.cdf(-self.Vanilla_d))
        #print self.x
        self.value = np.mean(self.x)

        print (self.value)
        return self.value

    def Accrual_Range_In (self):
         

         
          

        self.Upper_K = self.jsondata["Double_Accrual_Upper"]
        self.Lower_K = self.jsondata["Double_Accrual_Lower"]


        self.t_step = np.linspace(0.00274,self.T, 365)

        self.Lower_d1 = ((np.log(self.S0/self.Lower_K) + (self.b - self.sigma**2 / 2) * self.t_step)/(self.sigma * np.sqrt(self.t_step)))
        self.Lower_value = np.exp(-self.r * self.t_step)* ss.norm.cdf(self.Lower_d1)

        self.Upper_d1 = ((np.log(self.S0/self.Upper_K) + (self.b - self.sigma**2 / 2) * self.t_step)/(self.sigma * np.sqrt(self.t_step)))
        self.Upper_value = 1- (np.exp(-self.r * self.t_step)* ss.norm.cdf(self.Upper_d1))

        self.x = (self.Lower_value + self.Upper_value) -1
        #print self.x

        self.value = np.mean(self.x)

        print (self.value)
        return self.value

    def Accrual_Range_Out (self):
         

         
          

        self.Upper_K = self.jsondata["Double_Accrual_Upper"]
        self.Lower_K = self.jsondata["Double_Accrual_Lower"]

        self.t_step = np.linspace(0.00274,self.T, 365)
        #print self.t_step
        self.Lower_d1 = ((np.log(self.S0/self.Lower_K) + (self.b - self.sigma**2 / 2) * self.t_step)/(self.sigma * np.sqrt(self.t_step)))
        self.Lower_value = (np.exp(-self.r * self.t_step)* ss.norm.cdf(-self.Lower_d1))

        self.Upper_d1 = ((np.log(self.S0/self.Upper_K) + (self.b - self.sigma**2 / 2) * self.t_step)/(self.sigma * np.sqrt(self.t_step)))
        self.Upper_value = np.exp(-self.r * self.t_step)* ss.norm.cdf(-self.Upper_d1)

        self.x =  (self.Lower_value - self.Upper_value) + 1
        #print self.x
        self.value = np.mean(self.x)

        print (self.value)
        return self.value






if __name__ == "__main__":
    filename = "JSON_Sample_v2.json"
    with open(filename) as jsonfile:
        jsondata = json.load(jsonfile)

        Price = ClosedFormPricing()
        val = Price.getPrice(jsondata)




elapsed =  time.time() - start_time

print (elapsed)

