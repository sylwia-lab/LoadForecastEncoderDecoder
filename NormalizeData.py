# -*- coding: utf-8 -*-
"""
Spyder Editor

Load forecasting with 24 leight weighted ANNs
"""
import pandas as pd

lNormTypes=['Zero-Z-Score','Min-Max','Max']

class Feature:
  def __init__(self,name,minVal,maxVal,meanVal,stdDevVal):
    self.m_name=name
    self.m_minVal=minVal
    self.m_maxVal=maxVal
    self.m_meanVal=meanVal
    self.m_stdDevVal=stdDevVal

def getFeatureValues(dfTrainingData):
  dictValues={}
  lAllAttributes = list(dfTrainingData.columns)
  for colName in lAllAttributes:
    col = dfTrainingData[colName]
    minVal = col.min()
    maxVal = col.max()
    meanVal = col.mean()
    stdDevVal = col.std()
    #print(colName,minVal,maxVal,meanVal,stdDevVal)
    feature = Feature(colName,minVal,maxVal,meanVal,stdDevVal)
    dictValues[colName]=feature
  return dictValues
  
def getListOfAttributes(dfTrainingData):
  #"LoadPower_Cont","Temperature_Cont","DayOfWeek","PrevLoadPower_Cont","DeltaLoadPower_Cont","Timestamp","CurHour","Year","Day","Month","IsHoliday","IsHolidayPrev","IsHolidayNext","DailyMinTemp","DailyMinTempNextDay","DailyMaxTemp","DailyMinWetBulb","DailyMinWetBulbNextDay","DailyMaxWetBulb","Daylength","AvDailyTemp","Weight","MeanMaxTempLastDays","MeanMinTempLastDays","MeanMaxWetBulbLastDays","MeanMinWetBulbLastDays"
  #TimestampFirstHour
  #lConstantAttributes =["FilterDelta","FactorPrevLoadPower2Hour","RelTemp2PrevLoad","AvDailyTemp","Timestamp","TimestampFirstHour","CurHour","Year","Day","Month","DateHour","IsHoliday","IsHolidayPrev","IsHolidayNext","IsRegularDay","PrevIsRegularDay","PrevPrevIsRegularDay"]
  lConstantAttributes =["AvTempClass", "DayOfWeekUnNorm", "IdxDay", "LoadPower_Orig","SeasonalityIndex","Weight","KMeansLabels","IsTypicalDay","FilterDelta","FactorPrevLoadPower2Hour","RelTemp2PrevLoad","AvDailyTemp","Timestamp","TimestampFirstHour","CurHour","Year","Day","Month","DateHour","IsHoliday","IsHolidayPrev","IsHolidayNext","IsRegularDay","PrevIsRegularDay","PrevPrevIsRegularDay"]  

  lAllAttributes = list(dfTrainingData.columns)
  
  lAttributes=[]
  
  for entry in lAllAttributes:
    if entry in lConstantAttributes:
      continue
    else:
      lAttributes.append(entry)
      
  return lAttributes

class Normalizer:  
  def __init__(self,df,type):
    self.m_type = type
    self.m_df = df
    
def scaleInputData(dfTrainingData,type=0):
  
  lAttributes = getListOfAttributes(dfTrainingData)
  
  dictFeatureValues = getFeatureValues(dfTrainingData)
  
  #print(dictFeatureValues)
  
  dfTrainingDataNormalized = pd.DataFrame()
  
  dfTrainingDataNormalized = pd.concat([dfTrainingDataNormalized,dfTrainingData])
  
  for entry in lAttributes:
    if dictFeatureValues[entry].m_stdDevVal!=0:
      dfTrainingDataNormalized[entry]=(dfTrainingData[entry]-dictFeatureValues[entry].m_meanVal)/dictFeatureValues[entry].m_stdDevVal
    else:
      dfTrainingDataNormalized[entry]=(dfTrainingData[entry]-dictFeatureValues[entry].m_meanVal)
      
  
  return dfTrainingDataNormalized, dictFeatureValues
  
def scaleBackData(dfTrainingData,dictFeatureValues):
  lAttributes = getListOfAttributes(dfTrainingData)
  dfTrainingDataNotNormalized = pd.DataFrame()
  dfTrainingDataNotNormalized = pd.concat([dfTrainingDataNotNormalized,dfTrainingData])
  
  for entry in lAttributes:
    if dictFeatureValues[entry].m_stdDevVal!=0:
      dfTrainingDataNotNormalized[entry]=(dfTrainingData[entry]*dictFeatureValues[entry].m_stdDevVal+dictFeatureValues[entry].m_meanVal)
    else:
      dfTrainingDataNotNormalized[entry]=(dfTrainingData[entry]+dictFeatureValues[entry].m_meanVal)
  
  return dfTrainingDataNotNormalized
  
def scaleEvaluationInputData(dfTrainingData,dictFeatureValues):

  lAttributes = getListOfAttributes(dfTrainingData)
  
  dfTrainingDataNormalized = pd.DataFrame()
  
  dfTrainingDataNormalized = pd.concat([dfTrainingDataNormalized,dfTrainingData])
  for entry in lAttributes:
    if dictFeatureValues[entry].m_stdDevVal!=0:
      dfTrainingDataNormalized[entry]=(dfTrainingData[entry]-dictFeatureValues[entry].m_meanVal)/dictFeatureValues[entry].m_stdDevVal
    else:
      dfTrainingDataNormalized[entry]=(dfTrainingData[entry]-dictFeatureValues[entry].m_meanVal)
  
  
  return dfTrainingDataNormalized

def scaleEvaluationInputDataByMax(dfTrainingData,dictFeatureValues):

  lAttributes = getListOfAttributes(dfTrainingData)
  
  dfTrainingDataNormalized = pd.DataFrame()
  
  dfTrainingDataNormalized = pd.concat([dfTrainingDataNormalized,dfTrainingData])
  
  for entry in lAttributes:
    dfTrainingDataNormalized[entry]=(1/dictFeatureValues[entry].m_maxVal)*dfTrainingData[entry]

  return dfTrainingDataNormalized
  
def scaleEvaluationInputDataByDiffExtremeValues(dfTrainingData,dictFeatureValues):

  lAttributes = getListOfAttributes(dfTrainingData)
  
  dfTrainingDataNormalized = pd.DataFrame()
  
  dfTrainingDataNormalized = pd.concat([dfTrainingDataNormalized,dfTrainingData])
  
  for entry in lAttributes:
   
    diffVal = dictFeatureValues[entry].m_maxVal-dictFeatureValues[entry].m_minVal
    dfTrainingDataNormalized[entry]=(dfTrainingData[entry]-dictFeatureValues[entry].m_minVal)/diffVal

  return dfTrainingDataNormalized

def scaleScalar(unscaledScalar, feature):
  #dfTrainingDataNormalized[entry]=(dfTrainingData[entry]-dictFeatureValues[entry].m_meanVal)/dictFeatureValues[entry].m_stdDevVal
  if feature.m_stdDevVal!=0:
    return ((unscaledScalar-feature.m_meanVal)/feature.m_stdDevVal)
  else:
    return (unscaledScalar-feature.m_meanVal)
def scaleBackInputData(scaledData, feature):
  if feature.m_stdDevVal!=0:
    return ((scaledData*feature.m_stdDevVal)+feature.m_meanVal)
  else:
    return (scaledData+feature.m_meanVal)
def normalizeInputDataByMaxValue(dfTrainingData):
  lAttributes = getListOfAttributes(dfTrainingData)
  
  dictFeatureValues = getFeatureValues(dfTrainingData)
  
  dfTrainingDataNormalized = pd.DataFrame()
  
  dfTrainingDataNormalized = pd.concat([dfTrainingDataNormalized,dfTrainingData])
  
  for entry in lAttributes:
    dfTrainingDataNormalized[entry]=(1/dictFeatureValues[entry].m_maxVal)*dfTrainingData[entry]

  return dfTrainingDataNormalized, dictFeatureValues

def normalizeBackInputDataByMaxValue(scaledData, feature):
  return (scaledData*feature.m_maxVal)

def normalizeInputDataByDiffExtremeValues(dfTrainingData):
  lAttributes = getListOfAttributes(dfTrainingData)
  
  dictFeatureValues = getFeatureValues(dfTrainingData)
   
  dfTrainingDataNormalized = pd.DataFrame()
  
  dfTrainingDataNormalized = pd.concat([dfTrainingDataNormalized,dfTrainingData])
  
  for entry in lAttributes:
   
    diffVal = dictFeatureValues[entry].m_maxVal-dictFeatureValues[entry].m_minVal
    dfTrainingDataNormalized[entry]=(dfTrainingData[entry]-dictFeatureValues[entry].m_minVal)/diffVal

  return dfTrainingDataNormalized, dictFeatureValues

def normalizeBackInputDataByDiffExtremeValues(scaledData, feature):
  diffLoadPower = feature.m_minVal-feature.m_maxVal
  return (scaledData*diffLoadPower+feature.m_minVal)
  


