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

def calcDiff(dfAnyDay,dfForecastDay,lAttributes):
  if len(dfAnyDay.Timestamp)!=24 or len(dfForecastDay.Timestamp)!=24:
    return 1000
  diffAttributes=0
  for attribute in lAttributes:
    for idxHour in range(0,24):
      delta = dfAnyDay[attribute].iloc[idxHour]-dfForecastDay[attribute].iloc[idxHour]
      diffAttributes+=(delta*delta)
  return math.sqrt(diffAttributes)



def get_weight_for_temp(d_temp):
  d_weight = 1
  b = 0.82
  a = 0.171

  d_weight = a*d_temp + b
  return d_weight

def calc_weight_factor(attribute,dfAllDataNormalized):
  wf = 1
  fact = abs(max(dfAllDataNormalized[attribute])-min(dfAllDataNormalized[attribute]))
  if fact != 0:
    wf = 1/fact
  #print("WF", attribute,wf)
  return wf

def filter_closest_day(timestamp, is_holiday, is_holiday_prev):
    prev_timestamp = 0
    if is_holiday == is_holiday_prev:
        prev_timestamp = int(timestamp - 1*24*60*60)
    elif is_holiday == 1:
        prev_timestamp = int(timestamp - 7*24*60*60)
    else:
        prev_timestamp = int(timestamp - 3*24*60*60)
    return prev_timestamp

def calc_similarity(lAttributes,dfCurrent,dfPrevious):
  
  #lAttributes.append('IsHoliday')
  
  dfForecastDayRow0 = dfCurrent.iloc[0,:].copy()
  dfPrevDayRow0 = dfPrevious.iloc[0,:].copy()
  similarity = 0
  expFact=2.0
  weight = 1
  for attribute in lAttributes:
    av_attribute = (dfForecastDayRow0[attribute])
    weight=calc_weight_factor(attribute,dfPrevious)

    similarity += (weight*abs(dfPrevDayRow0[attribute]-av_attribute))**expFact
    
  return similarity


def filter_on_missing_previous_days(dfFilteredTrainingData, prev_days):
  
  dfFilteredTrainingData['DiffPrevDays'] = [0]*dfFilteredTrainingData.shape[0]
  
  for idx_day in range(prev_days,0,-1):
      print("idx_day", idx_day)
      delta = idx_day*24
      dfFilteredTrainingData['DiffPrevDays'] =  dfFilteredTrainingData['Timestamp'].diff(periods=delta)
  print(dfFilteredTrainingData['DiffPrevDays'])  
  return dfFilteredTrainingData

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


def avg_values(val, num_prev_days):
    if num_prev_days > 0:
        return val / num_prev_days
    else:
        return val

def process_val(is_in, cnt_prev_days, old_val, new_val):
    #print(is_in, cnt_prev_days, old_val, new_val)
    if is_in is False:
        return old_val
    if cnt_prev_days == 0:
        return new_val
    return (old_val + new_val)

def process_cnt_days(is_in, old_cnt):
    if is_in is False:
        return old_cnt
    return (old_cnt + 1)

    
def get_previous_days_avg(dfAllDataNormalized, dfFilteredTrainingData, prev_avg_days):
  #'Hour',
  lAttributesEncoder=['Temperature_Cont','WetBulb_Cont' ,'IsHoliday','DayOfWeek', 'Daylength', 'IsRegularDay',\
                      'DeltaTemp_Cont', 'PrevLoadPower_Cont']

  dfAllDataNormalized = dfAllDataNormalized.sort_values('Timestamp',ascending=True)
  dfFilteredTrainingData = dfFilteredTrainingData.sort_values('Timestamp',ascending=True)
      
  same_tables = False
  #throw away first 24 entries
  if dfAllDataNormalized.shape[0] == dfFilteredTrainingData.shape[0]:
      dfFilteredTrainingData = dfFilteredTrainingData.iloc[24:].copy()
      same_tables = True

  df_out = dfFilteredTrainingData.copy()
  df_out['CountPrevDays'] = [0]*df_out.shape[0]
  df_out['Timestamp'] = df_out['Timestamp']-24*60*60
  df_out['TimestampIsIn'] = [False]*df_out.shape[0]

 

  fake_row = dfAllDataNormalized.iloc[0]

  df_timestamps = pd.DataFrame()
  for idx_day in range(prev_avg_days,0,-1):
      print("idx_day", idx_day)
      delta_ts = idx_day*24*60*60
      
      df_timestamps['Timestamp'] = dfFilteredTrainingData['Timestamp']-delta_ts    
      l_timestamps = df_timestamps['Timestamp'].values
      print("get_previous_days: l_timestamps", len(l_timestamps))
      
      df_out['TimestampIsIn'] = df_timestamps['Timestamp'].isin(dfAllDataNormalized['Timestamp'])
      df_PrevDays_merged = pd.DataFrame()
      init = False
      for ts in l_timestamps:
          df_row = dfAllDataNormalized[dfAllDataNormalized['Timestamp'] == ts].copy()
          if df_row.shape[0] == 0:
              df_row = fake_row.copy()
              
          if init is False:
              df_PrevDays_merged = df_row.copy()
              init = True
          else:
              df_PrevDays_merged = pd.concat([df_PrevDays_merged,df_row], ignore_index=True, axis=0)
      
      #update attributes
      for entry in lAttributesEncoder:
          #print("dupa", entry, df_out.shape[0], df_PrevDays_merged.shape[0])
          df_out['Tmp'] = df_PrevDays_merged[entry].values
          df_out[entry] = \
              df_out.apply(lambda x: process_val(x.TimestampIsIn, x.CountPrevDays, x[entry], x.Tmp), axis=1)
      
      #update count days
      df_out['CountPrevDays'] = \
              df_out.apply(lambda x: process_cnt_days(x.TimestampIsIn, x.CountPrevDays), axis=1)
      
  
  for entry in lAttributesEncoder:
      df_out[entry] = \
              df_out.apply(lambda x: avg_values(x[entry], x.CountPrevDays), axis=1)  
  
  df_out = df_out[df_out.CountPrevDays > 0]
  df_out = df_out.drop(['CountPrevDays','TimestampIsIn', 'Tmp'], axis = 1) #
  df_out['FilterDelta'] = [0]*df_out.shape[0]

  #throw away first 24 entries
  if same_tables is True:
      dfAllDataNormalized = dfAllDataNormalized.iloc[24:].copy()
      

  print("df_out", df_out.shape, df_out['Year'].max(), df_out['Year'].min())
  print("dfAllDataNormalized", dfAllDataNormalized.shape, \
        dfAllDataNormalized['Year'].max(), dfAllDataNormalized['Year'].min())
  
  return dfAllDataNormalized, df_out


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
        


def calc_filter_delta(att_hist_data, att_forecast_day,weight, expFact):
    return (weight*abs(att_hist_data-att_forecast_day))**expFact


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




def get_hourly_sample_weights(lAttributes,dfForecastDay,dfFilteredTrainingData):
  print("dfFilteredTrainingData", dfFilteredTrainingData.shape[0]/24)
  res_array = np.zeros((int(dfFilteredTrainingData.shape[0]/24), 24))
  loc_lAttributes = ['Temperature_Cont','WetBulb_Cont']
  loc_lAttributes.extend(lAttributes)


  expFact=2.0
  weight = 1
  for idxHour in range(24):
      sub_df = dfFilteredTrainingData[dfFilteredTrainingData.CurHour == idxHour].copy()
      for attribute in loc_lAttributes:
          av_attribute = dfForecastDay.iloc[idxHour].loc[attribute]
          weight=calc_weight_factor(attribute,dfFilteredTrainingData)#sub_df)
          res_array[:,idxHour] += (weight*abs(sub_df[attribute]-av_attribute))**expFact
    
  res_array = (res_array/len(lAttributes))**(1.0/expFact)
  
  return res_array
  




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





