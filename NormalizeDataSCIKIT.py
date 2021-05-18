# -*- coding: utf-8 -*-
"""
Spyder Editor

Scaler used separately for each column
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

#,Timestamp,TimestampFirstHour,CurHour,IsHoliday,,IsHolidayPrev,IsHolidayNext,,DeltaLoadPower_Cont,Price_Cont,Losses_Cont,Congestion_Cont,,DayOfWeek,DailyMinTemp,DailyMaxTemp,DailyMinTempNextDay,DailyMinWetBulbNextDay,DailyMinWetBulb,DailyMaxWetBulb,MeanMaxTempLastDays,MeanMinTempLastDays,MeanMaxWetBulbLastDays,MeanMinWetBulbLastDays,Temperature_Cont,WetBulb_Cont,Sunset,Sunrise,Daylength,DeltaTemp_Cont,Year,Day,Month,Hour,DayOfYear,AvDailyTemp,IsRegularDay,,,PrevDeltaLoadPower_Cont,Weight
lNormTypes=['StandardScaler','MinMaxScaler','RobustScaler']
lConstantAttributes =["LoadPower_Orig", "SeasonalityIndex","Weight","IsTypicalDay","Sylwia3","Sylwia2","Sylwia1","FilterDelta","FactorPrevLoadPower2Hour","RelTemp2PrevLoad","AvDailyTemp","Timestamp","TimestampFirstHour","CurHour","Year","Day","Month","DateHour","IsHoliday","IsHolidayPrev","IsHolidayNext","IsRegularDay","PrevIsRegularDay","PrevPrevIsRegularDay"]  
#will be scaled on LoadPower_Cont
lPowerFeatures=['LoadPower_Cont','PrevPrevLoadPower_Cont','PrevLoadPower_Cont','PrevWeekLoad_Cont','PrevDayLoad_Cont']
  

class Feature:
  def __init__(self,name,minVal,maxVal,meanVal,stdDevVal,scaler):
    self.m_name=name
    self.m_minVal=minVal
    self.m_maxVal=maxVal
    self.m_meanVal=meanVal
    self.m_stdDevVal=stdDevVal
    self.m_scaler = scaler
    
def getFeatureValues(dfTrainingData,type,range):
  dictValues={}
  lAllAttributes = list(dfTrainingData.columns)
  for colName in lAllAttributes:
    col = dfTrainingData[colName]
    minVal = col.min()
    maxVal = col.max()
    meanVal = col.mean()
    stdDevVal = col.std()
    scaler = type2scaler(type,range)
    scaler = scaler.fit(col.values.reshape(-1,1))
    feature = Feature(colName,minVal,maxVal,meanVal,stdDevVal,scaler)
    dictValues[colName]=feature
  return dictValues
  

def getAttributesForNorm(dfTrainingData):
  lAllAttributes = list(dfTrainingData.columns)
  
  lAttributes=[]
  
  for entry in lAllAttributes:
    if entry in lConstantAttributes:
      continue
    else:
      lAttributes.append(entry)
      
  return lAttributes

def type2scaler(type,range):
  if (type =='MinMaxScaler'):
    return MinMaxScaler(feature_range=range)
  elif (type =='RobustScaler'):
    return RobustScaler()
  else:
    return StandardScaler()
    
def scaleInputData(dfTrainingData, type,range=(-1, 1)): 
  lAttributes = getAttributesForNorm(dfTrainingData)
  dfTrainingData_copy = dfTrainingData.copy()
  dictFeatureValues = getFeatureValues(dfTrainingData,type,range)
  #Overwrite some cols
  for col in lAttributes:
    dfTrainingData_copy[col] = dictFeatureValues[col].m_scaler.transform(dfTrainingData[col].values.reshape(-1,1))
  
  return dfTrainingData_copy, dictFeatureValues  

def scaleEvaluationInputData(dfEvalData,dictFeatureValues):
  lAttributes = getAttributesForNorm(dfEvalData)
  dfEvalData_copy = dfEvalData.copy()
  #Overwrite some cols
  for col in lAttributes:
    dfEvalData_copy[col]=dictFeatureValues[col].m_scaler.transform(dfEvalData[col].values.reshape(-1,1))
  
  return dfEvalData_copy

def scaleBackInputData(scaledVal, feature):
  return feature.m_scaler.inverse_transform(np.array([[scaledVal]]))[0][0]

def scaleScalar(toBeScaled, feature):
  df = pd.DataFrame()
  df['x']=[toBeScaled]
  return feature.m_scaler.transform(df['x'].values.reshape(-1,1))[0][0]
