# -*- coding: utf-8 -*-
"""
Spyder Editor

Load forecasting with stacked lstms
"""


import pandas as pd

import math
from sklearn.linear_model import Lasso

import NormalizeDataSCIKIT as normSCIKIT
import NormalizeData as norm
 
#NormalizeData,NormalizeDataSCIKIT_R
from statistics import mean, stdev
import numpy as np
import scipy.signal

import statsmodels.api as sm
import statsmodels.formula.api as smf

ZERO_THRE = 0.00001
START_YEAR = 2011


def getPredictionInput(predictionStart, predictionEnd, dfEvaluationDataNormalized):
  dfStartDay = dfEvaluationDataNormalized[(dfEvaluationDataNormalized.Year==predictionStart['Year'])]
  dfStartDay = dfStartDay[(dfStartDay.Month==predictionStart['Month'])]
  dfStartDay = dfStartDay[(dfStartDay.Day==predictionStart['Day'])]
  dfStartDay = dfStartDay[(dfStartDay.CurHour==predictionStart['CurHour'])]
  print(dfStartDay.shape[0])
  dfEndDay = dfEvaluationDataNormalized[(dfEvaluationDataNormalized.Year==predictionEnd['Year'])]
  dfEndDay = dfEndDay[(dfEndDay.Month==predictionEnd['Month'])]
  dfEndDay = dfEndDay[(dfEndDay.Day==predictionEnd['Day'])]
  dfEndDay = dfEndDay[(dfEndDay.CurHour==predictionEnd['CurHour'])]
  
  startTS = dfStartDay.iloc[0].Timestamp
  endTS = dfEndDay.iloc[0].Timestamp
  print(startTS,endTS,(endTS-startTS)/60*60)
  dfPredInput = dfEvaluationDataNormalized[(dfEvaluationDataNormalized.Timestamp>=startTS) & (dfEvaluationDataNormalized.Timestamp<=endTS)]
  
  return dfPredInput

def getWeatherDayType(dfForecastDay):
  if dfForecastDay.iloc[0].AvDailyTemp<65 and dfForecastDay.iloc[0].AvDailyTemp>25:#regular
    return "cRegular"  
  elif dfForecastDay.iloc[0].AvDailyTemp>=65:#hot
    return "cHot"  
  else:#cold
    return "cCold"

def is_regular_day(av_temp):
  if av_temp >= 65:
      return 0.1
  elif av_temp <= 25:
      return 0.5
  else:
      return 1
  #if av_temp<65 and av_temp>25:#regular
  #  return 1   
  #else:#cold
  #  return 0


def calc_weight_factor(attribute,dfAllDataNormalized):
  wf = 1
  fact = abs(max(dfAllDataNormalized[attribute])-min(dfAllDataNormalized[attribute]))
  if fact != 0:
    wf = 1/fact
  #print("WF", attribute,wf)
  return wf



def get_previous_days(dfAllDataNormalized, dfFilteredTrainingData, prev_days):
  
  dfAllDataNormalized = dfAllDataNormalized.sort_values('Timestamp',ascending=True)
  dfFilteredTrainingData = dfFilteredTrainingData.sort_values('Timestamp',ascending=True)
  
  print('dfFilteredTrainingData', dfFilteredTrainingData.shape[0])
  
  if dfAllDataNormalized.shape[0] == dfFilteredTrainingData.shape[0]:
      dfFilteredTrainingDataShort = dfFilteredTrainingData.iloc[:-24].copy()
      dfAllDataNormalizedShort = dfAllDataNormalized.iloc[24:].copy()
      print("get_previous_days: Same tables")
      print('dfFilteredTrainingDataShort', dfFilteredTrainingDataShort.shape[0])
      return dfAllDataNormalizedShort, dfFilteredTrainingDataShort
  else:
      dfFilteredTrainingDataShort = dfFilteredTrainingData.copy()

  l_timestamps = []
  
  for idx_day in range(prev_days,0,-1):
      print("idx_day", idx_day)
      delta_ts = idx_day*24*60*60
      #make a separate df with prev days
      df_timestamps = dfFilteredTrainingDataShort['Timestamp']-delta_ts    
      l_timestamps.extend(df_timestamps.values)
  
  print("get_previous_days: l_timestamps", len(l_timestamps))

              
  df_PrevDays_merged = pd.concat([dfAllDataNormalized[dfAllDataNormalized['Timestamp'].eq(x)] for x in l_timestamps], \
                                  ignore_index=True)
      
  df_PrevDays = df_PrevDays_merged.copy()
  df_PrevDays['FilterDelta'] = [0]*df_PrevDays.shape[0]
  

  print("df_PrevDays", df_PrevDays.shape, df_PrevDays['Year'].max(), df_PrevDays['Year'].min())
  print("dfAllDataNormalized", dfAllDataNormalized.shape, \
        dfAllDataNormalized['Year'].max(), dfAllDataNormalized['Year'].min())
  return dfAllDataNormalized, df_PrevDays


def insert_idx_day(dfFilteredTrainingData, df_PrevDays, prev_days):
  vec_idx_day_filtered = np.arange(int(dfFilteredTrainingData.shape[0]/24))
  print("dfFilteredTrainingData",dfFilteredTrainingData.shape)
  vec_idx_day_filtered = np.repeat(vec_idx_day_filtered,24)
  #print("vec_idx_day_filtered",vec_idx_day_filtered.shape)

  vec_idx_day_prev = np.repeat(vec_idx_day_filtered, prev_days)
  print("vec_idx_day_prev",vec_idx_day_prev.shape)
  print("df_PrevDays",df_PrevDays.shape)
  
  df_PrevDays['IdxDay']=vec_idx_day_prev
  dfFilteredTrainingData['IdxDay']=vec_idx_day_filtered
  return dfFilteredTrainingData, df_PrevDays
        


def filterDataByDayLengthTempRef(lAttributes, dfForecastDay, dfAllDataNormalized, dfAllDataNormalized_Prev, \
                                                 batch_size, \
                                                 num_last_data=0, prev_days=1):
    
  weatherDayType=getWeatherDayType(dfForecastDay)
  
  yearForecastDay = dfForecastDay.iloc[0].Year
  
  queryYearStart = yearForecastDay-1
  if dfForecastDay.iloc[0].IsHoliday == 1:
    queryYearStart = yearForecastDay-3 #3 2014
  if weatherDayType != "cRegular":
    queryYearStart = yearForecastDay-3 #3 2014
    
  
  queryDayStart = 1 #dfForecastDay.iloc[0].Day #1
  queryMonthStart = 1 #dfForecastDay.iloc[0].Month #1
  
  #Set to zero hour of forecast day
  tsFirstForecast = dfForecastDay[dfForecastDay.CurHour == 0].Timestamp.values[0]

  startTimestamp = dfAllDataNormalized[(dfAllDataNormalized.Year == queryYearStart) & \
                                       (dfAllDataNormalized.Month == queryMonthStart) & \
                                       (dfAllDataNormalized.Day == queryDayStart) & \
                                       (dfAllDataNormalized.CurHour==0)].iloc[0].Timestamp
  
  print("startTimestamp, queryYearStart, yearForecastDay", startTimestamp, queryYearStart, yearForecastDay)

  dfPreFilteredTrainingData = \
          dfAllDataNormalized[(dfAllDataNormalized.IsHoliday == dfForecastDay.iloc[0].IsHoliday)
                              ]
                                           
  dfPreFilteredTrainingData = dfPreFilteredTrainingData[(dfPreFilteredTrainingData.Timestamp>= startTimestamp) & \
                                                        (dfPreFilteredTrainingData.Timestamp < tsFirstForecast)].copy()

  
    

  dfPreFilteredTrainingData['FilterDelta'] = [0 for i in range(len(dfPreFilteredTrainingData['Timestamp']))]

  

  dfForecastDayRow0 = dfForecastDay.iloc[0,:]
      
  expFact=2.0
  weight = 1
  for attribute in lAttributes:
      av_attribute = (dfForecastDayRow0[attribute])
      weight=calc_weight_factor(attribute,dfPreFilteredTrainingData)
      dfPreFilteredTrainingData['FilterDelta'] += (weight*abs(dfPreFilteredTrainingData[attribute]-av_attribute))**expFact
      

  dfPreFilteredTrainingData['FilterDelta'] = dfPreFilteredTrainingData['FilterDelta'].values
  dfPreFilteredTrainingData['FilterDelta'] = (dfPreFilteredTrainingData['FilterDelta']/(len(lAttributes)))**(1.0/expFact)

  
  tsWithMaxStd=int(batch_size)
  dfPreFilteredTrainingData = dfPreFilteredTrainingData.sort_values(['FilterDelta','Timestamp'],ascending=[True, False]).copy()
  dfFilteredTrainingData = dfPreFilteredTrainingData[0:tsWithMaxStd*24]
  
  dfFilteredTrainingData = dfFilteredTrainingData.sort_values(['Timestamp'],ascending=True)
  l_idx_prev_days = dfFilteredTrainingData['IdxDay'].unique()
  
  #dfAllDataNormalized, df_PrevDays = get_previous_days(dfAllDataNormalized, dfFilteredTrainingData, 1)
  df_PrevDays = pd.concat([dfAllDataNormalized_Prev[dfAllDataNormalized_Prev['IdxDay'].eq(x)] for x in l_idx_prev_days], \
                                  ignore_index=True)

  
  #dfFilteredTrainingData, df_PrevDays = insert_idx_day(dfFilteredTrainingData, df_PrevDays, prev_days)
  
  print("Shapes", dfFilteredTrainingData.shape[0],df_PrevDays.shape[0])
  #zero because it is not regarded while weighting the training data
  df_PrevDays['FilterDelta'] = [0]*df_PrevDays.shape[0]
  
  #Number of rows    
  print("filterDataByDayLengthTempRef_LSTM Number training data days: ",dfFilteredTrainingData.shape[0]/24)

  print("df_PrevDays",df_PrevDays.shape[0],'dfFilteredTrainingData',dfFilteredTrainingData.shape[0])
  if df_PrevDays.shape[0] != dfFilteredTrainingData.shape[0]*prev_days:
    raise ValueError("Unequal rows!")
  if df_PrevDays.shape[1] != dfFilteredTrainingData.shape[1]:
    raise ValueError("Unequal cols!")
  if dfFilteredTrainingData.shape[0] != batch_size*24:
    raise ValueError('Filtered data does not match expected shape!')
    
  for idxHour in range(24):
      if dfFilteredTrainingData[dfFilteredTrainingData.CurHour == idxHour].shape[0] != batch_size:
          print(dfFilteredTrainingData[dfFilteredTrainingData.CurHour == idxHour].shape[0])
          raise ValueError('Filtered data does not match expected shape for hour %d!' % idxHour)
  
  
  return dfFilteredTrainingData, df_PrevDays



def calc_free_day(day_of_week, is_holiday):
  #not a weekend
  if is_holiday == 1 and (day_of_week != 6 and day_of_week!=0):
    return 1
  else:
    return 0


    
def getDataFrameNormalized(pathTrainingData, pathEvaluationData):
  #Read training data in data frame, R export
  dfTrainingData = pd.read_csv(pathTrainingData,delimiter=",",encoding="utf-8-sig")
  dfEvaluationData = pd.read_csv(pathEvaluationData,delimiter=",",encoding="utf-8-sig")
  dfTrainingData = dfTrainingData[dfTrainingData.Year>=START_YEAR]
  
  dfTrainingData['FreeDay'] = dfTrainingData['DayOfWeek']
  dfTrainingData['FreeDay'] = dfTrainingData.apply(lambda x: calc_free_day(x.DayOfWeek, x.IsHoliday), axis=1)
  dfTrainingData['LoadPower_Orig'] = dfTrainingData['LoadPower_Cont']
  
  dfEvaluationData['FreeDay'] = dfEvaluationData['DayOfWeek']
  dfEvaluationData['FreeDay'] = dfEvaluationData.apply(lambda x: calc_free_day(x.DayOfWeek, x.IsHoliday), axis=1)
  dfEvaluationData['LoadPower_Orig'] = dfEvaluationData['LoadPower_Cont']
  
  tsTraining = int(dfTrainingData[(dfTrainingData.Year==START_YEAR) & (dfTrainingData.Month==1) & \
                                  (dfTrainingData.Day==1) & (dfTrainingData.CurHour==0)].iloc[0].Timestamp)
  
  dfTrainingData = dfTrainingData[(dfTrainingData.Timestamp>=tsTraining)]
  
  #Add day id
  dfTrainingData = add_day_idx(dfTrainingData)
  dfEvaluationData = add_day_idx(dfEvaluationData, start_idx = (dfTrainingData.shape[0]/24))
  
  
  dfAllData = pd.concat([dfTrainingData,dfEvaluationData],ignore_index=True)
  dfAllData = dfAllData.sort_values('Timestamp',ascending=True)
  #dfAllData = decompose_signal(dfAllData)
  
  dfEvaluationData = dfAllData[dfAllData.Year >= 2017]
  dfTrainingData = dfAllData[dfAllData.Year < 2017]
  
  #dfTrainingData = remove_yearly_trend(dfTrainingData)
 
  #Normalize with training data
  range=(-1,1)
  #dfTrainingDataNormalized, dictFeatures = normSCIKIT.scaleInputData(dfTrainingData,'MinMaxScaler',range)# 'StandardScaler' MinMaxScaler
  #dfEvaluationDataNormalized = normSCIKIT.scaleEvaluationInputData(dfEvaluationData,dictFeatures)
  
  dfTrainingDataNormalized, dictFeatures = norm.scaleInputData(dfTrainingData)# 'MinMaxScaler'
  dfEvaluationDataNormalized = norm.scaleEvaluationInputData(dfEvaluationData,dictFeatures)
  
  #Concatenate for the ease of the usage
  listAllData = [dfTrainingDataNormalized,dfEvaluationDataNormalized]
  dfAllDataNormalized = pd.concat(listAllData,ignore_index=True)
  
  #dfAllDataNormalized, dictFeatures = normSCIKIT.scaleInputData(dfAllData,'MinMaxScaler',range)
  #dfAllDataNormalized, dictFeatures = norm.scaleInputData(dfAllData)
  
  dfAllDataNormalized = dfAllDataNormalized.sort_values('Timestamp',ascending=True)
  #201118
  dfAllDataNormalized['IsRegularDay'] = dfAllDataNormalized['AvDailyTemp'].apply(is_regular_day)
  dfAllDataNormalized['ValidForTraining']=[1]*dfAllDataNormalized.shape[0]
  
  #get prev days df, since everything conditioned on prev day
  dfAllDataNormalized, df_PrevDays = get_previous_days(dfAllDataNormalized, dfAllDataNormalized.copy(), 1)
  #set day index
  dfAllDataNormalized, df_PrevDays = insert_idx_day(dfAllDataNormalized, df_PrevDays, 1)
  
  
  #dfAllDataNormalized.to_csv('C:\\Data\\DB-LF_ANN_190915\\ANN_Python\\NYISO_GitHub\\Data\\TrainingDataNormAll.csv',float_format='%.3f')
  
  return dfAllData, dfAllDataNormalized, df_PrevDays, dictFeatures

def add_day_idx(dfTrainingData, start_idx=0):
    dfTrainingData = dfTrainingData.sort_values(['Timestamp'],ascending=True)
    vec_idx_day_filtered = np.arange(dfTrainingData.shape[0]/24) + start_idx
    #print("vec_idx_day_filtered",vec_idx_day_filtered.shape)
    vec_idx_day_filtered = np.repeat(vec_idx_day_filtered,24)
    #print("vec_idx_day_filtered",vec_idx_day_filtered.shape)

    dfTrainingData['DayID']=vec_idx_day_filtered
    return dfTrainingData

def get_date(year, month, day):
    return str(int(year)) + '-' + str(int(month)) + '-' + str(int(day))





