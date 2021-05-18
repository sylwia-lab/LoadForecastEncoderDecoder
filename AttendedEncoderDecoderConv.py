# -*- coding: utf-8 -*-
# Seed value
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow
tensorflow.random.set_seed(seed_value)
#tensorflow.compat.disable_eager_execution()

# 5. Configure a new global `tensorflow` session

session_conf = tensorflow.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tensorflow.compat.v1.Session(graph=tensorflow.compat.v1.get_default_graph(), config=session_conf)

tensorflow.compat.v1.keras.backend.set_session(sess)

import pandas as pd
import tensorflow.keras.models as tf_models

from tensorflow.keras.layers import Activation,Dense, BatchNormalization,Input, Add
from tensorflow.keras.layers import LSTM, SimpleRNN,TimeDistributed,GRU,Bidirectional,Conv1D,Conv2D,\
MaxPooling1D,Flatten,Reshape,MaxPooling2D,AveragePooling2D,GlobalAveragePooling1D,GlobalMaxPooling1D,\
ConvLSTM2D, Concatenate, concatenate, dot, Dot, RepeatVector, Multiply, Permute, Softmax, \
Cropping1D

from tensorflow.keras.regularizers import L1L2


from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K


#import os.path
from os import path
import glob 
import os

import InputDataFilter as flt


#import NormalizeDataSCIKIT as norm #normSCIKIT
import NormalizeData as norm


import time
import math
from statistics import mean

#from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import gc

from tensorflow.keras.layers import ELU, PReLU
from SimplifiedAttention import SimplifiedAttention as GlobalAttention

#tensorflow.compat.v1.disable_eager_execution()




lAttributesFilter=['LoadPower_Cont23','Daylength',\
                   'DailyMaxTemp','DailyMinTemp','DailyMinTempNextDay',\
                   'DailyMaxWetBulb','DailyMinWetBulb','DailyMinWetBulbNextDay',\
                   'IsHolidayNext','IsHolidayPrev','IsHoliday',\
                   'MeanMaxTempLastDays','MeanMinTempLastDays',\
                   'MeanMaxWetBulbLastDays','MeanMinWetBulbLastDays']
  
                                  
                   

lAttributesDecoder=['Temperature_Cont','WetBulb_Cont','IsHoliday','DayOfWeek', 'Daylength', 'IsRegularDay','Hour',\
                    'DeltaTemp_Cont','PrevLoadPower_Cont'] #'PrevWeekLoad_Cont'
lAttributesEncoder=['Temperature_Cont','WetBulb_Cont','IsHoliday','DayOfWeek','Daylength', 'IsRegularDay','Hour',\
                    'DeltaTemp_Cont','PrevLoadPower_Cont']#, 'PrevPrevLoadPower_Cont'] #'PrevWeekLoad_Cont'

#NumLSTMCells 48, batch_size 96 #0.01
#"NYC","CENTRL","CAPITL","DUNWOD","GENESE","HUD_VL","LONGIL","MHK_VL","NORTH","WEST", "MILL_WD"
dictParams={'ZoneName': 'NYC',\
            'ANNTarget':'LoadPower_Cont',\
            'batch_size':96,'NumHoursBack':24,'NumHoursAhead':24,\
            'CorrThre':0.3, \
            'ActivationFunc':'tanh','TransferLearning':'True','RangeScale2':(0,1),\
            'EarlyStoppingThre':1e-5,'LearningRate':0.001,\
            'NumEpochs':50,'NumEpochsTransferLearning':50,'lAttributesFilter':lAttributesFilter,\
            'lAttributesDecoder':lAttributesDecoder,\
            'lAttributesEncoder':lAttributesEncoder,\
            'UseInputLayerNorm':False,'NumLSTMCells':48,\
            'UseAttention':1, 'Debug':0, 'ReverseInputEncoder':0, \
            'NumPrevDays':1, \
            'UseConvFilterEnc': 1, 'UseConvFilterDec': 1, \
            'KernelSizeEnc':1, 'KernelSizeDec':1}


    

#LoadPower_Cont
class EarlyStoppingByLossVal(Callback):
  def __init__(self, monitor='loss', mode='min',value=0.0001, verbose=0):
    super(Callback, self).__init__()
    self.monitor = monitor
    self.value = value
    self.verbose = verbose
    self.mode=mode

  def on_epoch_end(self, epoch, logs={}):
    current = logs.get(self.monitor)
    if current is None:
      print("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

    if current < self.value:
      if self.verbose > 0:
        print("Epoch %05d: early stopping THR" % epoch)
      self.model.stop_training = True
      
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpointSmallestMSE = EarlyStoppingByLossVal(monitor='loss',mode='min', value=0.0001, verbose=1)#value=0.0001
callbacks_list = [checkpointSmallestMSE]   

class EncoderDecoderRNN:
  def __init__(self, lFeaturesEncoder,lFeaturesDecoder,lTargetNodes,\
               iHour,iNumHoursBack,iNumHoursAhead):
    self.m_lFeaturesEncoder=lFeaturesEncoder
    self.m_lFeaturesDecoder=lFeaturesDecoder
    self.m_lTargetNodes=lTargetNodes
    self.m_ModelFileName = ""
    self.m_iHour = iHour
    self.m_iHoursBack = iNumHoursBack
    self.m_iHoursAhead = iNumHoursAhead
    self.m_mu = 0
    self.m_sigma = 0
    
  #It returns only one entry for inference
  def getInputNodesEvaluation(self,dfForecastDay,dfPrevDay):
    #print("getInputNodesEvaluation",self.m_iHour)
    
    diffHours = self.m_iHour - self.m_iHoursBack
    if self.m_iHour == 0: #encoder input in dfPrevDay
        start_hour_prev = 23-self.m_iHoursBack+1
        end_hour_prev = 23        
        df_data_encoder = dfPrevDay[(dfPrevDay.CurHour <= end_hour_prev) & \
                                   (dfPrevDay.CurHour >= start_hour_prev)].copy()

    elif  self.m_iHour > 0 and diffHours < 0: #encoder input in dfPrevDay and dfForecastDay
        start_hour_prev = 23 + diffHours + 1
        end_hour_prev = 23
        
        start_hour_curr = 0
        end_hour_curr = self.m_iHour-1
        
        df_data_encoder_prev = dfPrevDay[(dfPrevDay.CurHour <= end_hour_prev) & \
                                   (dfPrevDay.CurHour >= start_hour_prev)].copy()
        df_data_encoder_curr = dfForecastDay[(dfForecastDay.CurHour <= end_hour_curr) & \
                                   (dfForecastDay.CurHour >= start_hour_curr)].copy()
        df_data_encoder = pd.concat([df_data_encoder_prev,df_data_encoder_curr],ignore_index=True)
        
    else: #encoder input in dfForecastDay
        start_hour_prev = self.m_iHour-self.m_iHoursBack
        end_hour_prev = self.m_iHour-1
        
        df_data_encoder = dfForecastDay[(dfForecastDay.CurHour <= end_hour_prev) & \
                                   (dfForecastDay.CurHour >= start_hour_prev)].copy()
            
    
    df_data_decoder = dfForecastDay[(dfForecastDay.CurHour == self.m_iHour)].copy()
        
      
    df_data_encoder = df_data_encoder.sort_values(['Timestamp'],ascending=True)
    df_data_decoder = df_data_decoder.sort_values(['Timestamp'],ascending=True)
    
    df_encoder = pd.DataFrame()
    df_decoder = pd.DataFrame()
    
    df_encoder['CurHour'] = df_data_encoder['CurHour']
    
    for feature in self.m_lFeaturesDecoder:
      df_decoder[feature]=df_data_decoder[feature]
      
    for feature in self.m_lFeaturesEncoder:
      df_encoder[feature]=df_data_encoder[feature]
      
    #reverse input encoder
    #if dictParams['ReverseInputEncoder'] == 1:
    #    df_encoder = df_encoder.sort_values(['CurHour'],ascending=False)
    #print(df_encoder)
    #I need 2D Tensor => matrix
    #dimensions: number_days x number_hours x number features
    #npResArray = np.empty((num_days,num_time_instances,num_features),float)
    v_2DTensorsEncoder=[]
    v_2DTensorsDecoder=[]

    #It converts the data frame to a matrix, 2D tensor, rows of df are rows of a matrix
    enc = df_encoder.iloc[:,1:].to_numpy()    
    dec = df_decoder.iloc[:,:].to_numpy()
    
    if dictParams['UseConvFilterEnc'] == 1:
        enc = np.transpose(enc)
    if dictParams['UseConvFilterDec'] == 1:
        dec = np.transpose(dec)
        
    #print(timeSeriesID,a.shape)
    v_2DTensorsEncoder.append(enc)
    v_2DTensorsDecoder.append(dec)

    #npResArray = np.array(v_2DTensors)
    
    npResArrayEncoder = np.array(v_2DTensorsEncoder)
    npResArrayDecoder = np.array(v_2DTensorsDecoder)
 
    #print("npResArrayPrevOutDecoder",npResArrayPrevOutDecoder.shape)
    #print("Inference data shape: ", npResArrayPrevOutDecoder.shape, npResArrayPrevOutDecoder.shape)
    return [npResArrayEncoder,npResArrayDecoder]
    
  #training features + time series values from hour h-24  
  def getInputOutputNodesTraining(self,df_curr_data,df_prev_data, batch_size=dictParams['batch_size'], \
                                  num_prev_days=dictParams['NumPrevDays']):
      
    if df_curr_data.shape[0]*num_prev_days != df_prev_data.shape[0]:
        raise ValueError('Unequal size of frames!')
    else:
        print("getInputOutputNodesTraining(): Equal frames!")
      
    diffHours = self.m_iHour - self.m_iHoursBack
    if 1==1:#self.m_iHour == 0: #encoder input in dfPrevDay
        start_hour_prev = 23-self.m_iHoursBack+1
        end_hour_prev = 23        
        dfTrainingPrev = df_prev_data[(df_prev_data.CurHour <= end_hour_prev) & \
                                   (df_prev_data.CurHour >= start_hour_prev)].copy()
    #h=3,b=4,diff=-1; hour = 3, diff = -1
    elif  self.m_iHour > 0 and diffHours < 0: #encoder input in dfPrevDay and dfForecastDay
        start_hour_prev = 23 + diffHours + 1
        end_hour_prev = 23
        
        start_hour_curr = 0
        end_hour_curr = self.m_iHour-1
        
        dfTrainingPrev_1 = df_prev_data[(df_prev_data.CurHour <= end_hour_prev) & \
                                   (df_prev_data.CurHour >= start_hour_prev)].copy()
        dfTrainingPrev_2 = df_curr_data[(df_curr_data.CurHour <= end_hour_curr) & \
                                   (df_curr_data.CurHour >= start_hour_curr)].copy()
        dfTrainingPrev = pd.concat([dfTrainingPrev_1,dfTrainingPrev_2],ignore_index=True)
        
    else: #encoder input in current day
        start_hour_prev = self.m_iHour-self.m_iHoursBack
        end_hour_prev = self.m_iHour-1
        
        dfTrainingPrev = df_curr_data[(df_curr_data.CurHour <= end_hour_prev) & \
                                      (df_curr_data.CurHour >= start_hour_prev)].copy()

    start_hour_cur = self.m_iHour
    end_hour_cur = self.m_iHour + self.m_iHoursAhead -1
      

    dfTrainingCur = df_curr_data[(df_curr_data.CurHour <= end_hour_cur) & \
                                    (df_curr_data.CurHour >= start_hour_cur)].copy()
    
    
    #if dfTrainingPrev.shape[0] != dfTrainingCur.shape[0]:
    #    raise ValueError('Unequal size of frames: dfTrainingCur = %d / dfTrainingPrev = %d!' % (dfTrainingPrev.shape[0], dfTrainingCur.shape[0]))
    #else:
    #    print("getInputOutputNodesTraining(): Equal frames! dfTrainingCur / dfTrainingPrev")
    
      
    l_idx_day = dfTrainingPrev.IdxDay.unique()
    df_encoder = pd.DataFrame()
    df_decoder = pd.DataFrame()
    df_encoder['IdxDay'] = dfTrainingPrev['IdxDay']
    df_encoder['CurHour'] = dfTrainingPrev['CurHour']
    df_encoder['ANNTarget'] = dfTrainingPrev[dictParams['ANNTarget']]
    

    df_decoder['IdxDay'] = dfTrainingCur['IdxDay']
    df_decoder['CurHour'] = dfTrainingCur['CurHour']
    df_decoder['ANNTarget'] = dfTrainingCur[dictParams['ANNTarget']]
    
      
    for feature in self.m_lFeaturesDecoder:
      df_decoder[feature]=dfTrainingCur[feature]
      
    for feature in self.m_lFeaturesEncoder:
      df_encoder[feature]=dfTrainingPrev[feature]
      
    #I need 3D Tensor time sequence [[[1,2,3],..,[6,7,8]],[]]
    #dimensions: number_days x number_hours x number features
    #npResArray = np.empty((num_days,num_time_instances,num_features),float)
    
    
    l_input_encoder = []

    l_input_decoder = [[] for i in range(24)]
    l_output = [[] for i in range(24)]


    for idx_day in l_idx_day:
        
      df_idx_day_encoder = df_encoder[df_encoder['IdxDay']==idx_day]
      df_idx_day_decoder = df_decoder[df_decoder['IdxDay']==idx_day]

      for idx_single_day in range(0, df_idx_day_encoder.shape[0], 24):
          df_day_encoder = df_idx_day_encoder.iloc[idx_single_day:(idx_single_day+24)]
          df_day_decoder = df_idx_day_decoder.iloc[idx_single_day:(idx_single_day+24)]
      
      
          #It converts the data frame to a matrix, 2D tensor, rows of df are rows of a matrix
          input_encoder = df_day_encoder.iloc[:,3:].to_numpy() #without IdxDay + cur_hour + loadpower
      
          if dictParams['UseConvFilterEnc'] == 1:
              input_encoder = np.transpose(input_encoder)
      
          l_input_encoder.append(input_encoder)
      
          for idx_hour in range(0,24):
              df_hour_decoder = df_day_decoder[df_day_decoder.CurHour == idx_hour]
              input_decoder = df_hour_decoder.iloc[:,3:].to_numpy() #without IdxDay + cur_hour + loadpower
              output_decoder = df_hour_decoder.iloc[:,2].to_numpy() #LoadPower_Cont as target
              #print("l_input_decoder[idx_hour]",idx_hour, len(l_input_decoder[idx_hour]))
              #print("l_output[idx_hour]",idx_hour, len(l_output[idx_hour]))
              if dictParams['UseConvFilterDec'] == 1:
                  input_decoder = np.transpose(input_decoder)
                  #print("input_decoder", input_decoder.shape)
              
              l_input_decoder[idx_hour].append(input_decoder)
              l_output[idx_hour].append(output_decoder)
      
    
    l_input_encoder_decoder = []
    l_output_encoder_decoder = []
    l_input_encoder_decoder.append(np.array(l_input_encoder))
    for idx_hour in range(0,24):
        l_input_encoder_decoder.append(np.array(l_input_decoder[idx_hour]))
        l_output_encoder_decoder.append(np.array(l_output[idx_hour]))
    
    

    return l_input_encoder_decoder, \
        l_output_encoder_decoder 
    
  
  def getInputSizeEncoder(self):
    numInputNodes = len(self.m_lFeaturesEncoder)
    return numInputNodes
  def getInputSizeDecoder(self):
    numInputNodes = len(self.m_lFeaturesDecoder)
    return numInputNodes    


def printListOfResults(pathResults, listOfResults):
  df = pd.DataFrame(listOfResults, columns=['Timestamp','IdxDay','Year', 'Month','Day', 'Hour','IsHoliday', 'DayOfWeek','IsRegularDay','Temperature','Real','Predicted','Deviation'])
  df = df[df.Year > 2016]
  df.to_csv(pathResults)
  #print(listOfResults)

def idxToMonth(idx):
  if idx == 1:
    return "January"
  elif idx == 2:
    return "February"
  elif idx == 3:
    return "March"
  elif idx == 4:
    return "April"
  elif idx == 5:
    return "Mai"
  elif idx == 6:
    return "June"
  elif idx == 7:
    return "July"
  elif idx == 8:
    return "August"
  elif idx == 9:
    return "September"
  elif idx == 10:
    return "October"
  elif idx == 11:
    return "November"
  elif idx == 12:
    return "December"
  else:
    return "Undef month"  
def printFormatedResults(fileName):
  df = pd.read_csv(fileName)
  dictMonthRes = {}
  for idx in range(1,13):
    dfMonth = df[(df.Month==idx)]
    dfMonthWorkingday = dfMonth[(dfMonth.IsHoliday==0)]
    dfMonthHoliday = dfMonth[(dfMonth.IsHoliday==1)]
    meanHoliday=0
    meanWorkday=0
    if len(dfMonthWorkingday['Deviation'])>0:
      meanWorkday=mean(dfMonthWorkingday['Deviation'])
    if len(dfMonthHoliday['Deviation'])>0:
      meanHoliday=mean(dfMonthHoliday['Deviation'])
    numRows = len(dfMonth['Deviation'])
    if numRows==0:
      continue
    dictMonthRes[idxToMonth(idx)]={'MAPE Month': mean(dfMonth['Deviation']),'MAPE Workingday':meanWorkday,'MAPE Holiday':meanHoliday}
    print(idxToMonth(idx),mean(dfMonth['Deviation']),"Workday",meanWorkday,"Holiday",meanHoliday)
  #print(dictMonthRes)

def create_training_model_with_attention(annModel, reg, file_name, num_prev_days=dictParams['NumPrevDays']): 
  num_features_encoder = annModel.getInputSizeEncoder()
  num_features_decoder = annModel.getInputSizeDecoder()
  
  activFunc=dictParams['ActivationFunc']#'tanh'#'sigmoid'#'relu'#'selu'tanh 'softsign'

  initializer=tensorflow.keras.initializers.glorot_uniform()#no seed, standard init on dense layer
  num_lstm_units = dictParams['NumLSTMCells']
  
  num_timesteps_before = annModel.m_iHoursBack*num_prev_days
  num_timesteps_after = annModel.m_iHoursAhead
  
  # Define an input sequence and process it, 
  # only 3rd dim must fit, actually: batch_size x num_timesteps_before x num_features
  #None => num_timesteps_before
  if dictParams['UseConvFilterEnc'] == 0:
      encoder_inputs = Input(shape=(None, num_features_encoder), name="encoder_input")
      filter_layer_encoder = encoder_inputs
  else:
      #num_features: 9 - 2 +1 => 8 features
      encoder_inputs = Input(shape=(num_features_encoder, num_timesteps_before), name="encoder_input")
      filter_layer_encoder = Conv1D(filters=num_timesteps_before, kernel_size=dictParams['KernelSizeEnc'], \
                        padding='valid', name='filter_layer_encoder')(encoder_inputs)
      print("filter_layer_encoder", filter_layer_encoder.shape)
      num_features = num_features_encoder - dictParams['KernelSizeEnc'] + 1
      #transpose input back
      filter_layer_encoder = Reshape((num_timesteps_before, num_features))(filter_layer_encoder)
      print("filter_layer_encoder", filter_layer_encoder.shape)

  encoder_0 = LSTM(num_lstm_units, activation=activFunc,kernel_initializer=initializer, \
                  name='encoder_lstm_0', return_sequences=True, bias_regularizer=reg)(filter_layer_encoder)
  encoder = LSTM(num_lstm_units, activation=activFunc,kernel_initializer=initializer, \
                 return_sequences=True, name='encoder_lstm_2', return_state=True, bias_regularizer=reg)
  encoder_outputs, state_h, state_c = encoder(encoder_0)

  print('encoder_outputs', encoder_outputs.shape)
  
  # We need the complete output for the attention model
  initial_states = [state_h, state_c]
  
  #decoder_inputs = Input(shape=(num_timesteps_after, num_features_decoder),name="decoder_input")
  
  l_decoder_inputs = []
  l_decoder_outputs = []
  
  for time_instance in range(num_timesteps_after):
      #create lstm layer
      if dictParams['UseConvFilterDec'] == 1:
          input_name = "decoder_input_" + str(time_instance)
          input_lstm = Input(shape=(num_features_decoder, 1),name=input_name)
          l_decoder_inputs.append(input_lstm)
          
          filter_layer_name = "dec_filter_layer_" + str(time_instance)
          filter_layer_dec = Conv1D(filters=1, kernel_size=dictParams['KernelSizeDec'], \
                        padding='valid', name=filter_layer_name)(input_lstm)
          num_features = num_features_decoder - dictParams['KernelSizeDec'] + 1
          filter_layer_dec = Reshape((1,num_features))(filter_layer_dec)
          
          lstm_layer_name = "decoder_lstm_" + str(time_instance)
          decoder_lstm = LSTM(num_lstm_units, activation=activFunc,kernel_initializer=initializer, \
                          return_sequences=True,name=lstm_layer_name, return_state=True, \
                          bias_regularizer=reg)
          decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(filter_layer_dec, \
                                                              initial_state=initial_states)
      
      else:
          input_name = "decoder_input_" + str(time_instance)
          input_lstm = Input(shape=(1, num_features_decoder),name=input_name)
          l_decoder_inputs.append(input_lstm)
          lstm_layer_name = "decoder_lstm_" + str(time_instance)
          decoder_lstm = LSTM(num_lstm_units, activation=activFunc,kernel_initializer=initializer, \
                          return_sequences=True,name=lstm_layer_name, return_state=True, \
                          bias_regularizer=reg)
          decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(input_lstm, \
                                                              initial_state=initial_states)
      #execute attention
      if dictParams['UseAttention'] == 1:
          #num_timestamps_filtered = dictParams['HoursEncoder']
          #flat_encoder = Reshape((num_timesteps_before*num_lstm_units,))(encoder_outputs)
          #flat_decoder = Reshape((num_lstm_units,))(decoder_outputs)
          #concat_concat_encoder_decoder = Concatenate(axis=1)([flat_encoder,flat_decoder])
          attention_layer_name = 'attention_layer_' + str(time_instance)
          attention_layer = GlobalAttention(num_lstm_units,num_timesteps_before,\
                                            initializer,layer_name=attention_layer_name)
          attended_decoder = attention_layer(encoder_outputs, decoder_outputs)
          dense_name = "decoder_output_" + str(time_instance)
          decoder_dense = Dense(1, name = dense_name)
          decoder_output = decoder_dense(attended_decoder)
          #decoder_state_c = Multiply()([attended_decoder, decoder_state_c])
          #connect attention result with the next layer
          initial_states = [attended_decoder, decoder_state_c]
      else:
          dense_name = "decoder_output_" + str(time_instance)
          decoder_dense = Dense(1, name = dense_name)
          decoder_output = decoder_dense(decoder_outputs)
          #connect attention result with the next layer
          initial_states = [decoder_state_h, decoder_state_c]
      
      l_decoder_outputs.append(decoder_output)
      
  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  ann_model = Model([encoder_inputs, l_decoder_inputs], l_decoder_outputs) #decoder_outputs
  
  with open(file_name,'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    ann_model.summary(print_fn=lambda x: fh.write(x + '\n'))
  
  # Set every layer to be trainable:
  for layer in ann_model.layers:
   layer.trainable = True
  
  learningRate = dictParams['LearningRate'] #0.001

  #Adam #1e-08 beta_2 = 0.999
  adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, \
                         decay=0.0) #0.999  , clipvalue=0.5 amsgrad=True
  #clipnorm=1.
  #clipvalue=0.5
  #adam = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  
  
  ann_model.compile(loss='mean_squared_error', metrics=['mean_squared_error'],\
                    optimizer=adam)#metrics=['mean_squared_error']    
  print("Model compiled successfully!")
  #print("ann_model.input", ann_model.input)
  #print("ann_model.output", ann_model.output)
  return ann_model

def create_inference_encoder_model(training_model):

  encoder_inputs = training_model.get_layer('encoder_input').input
  
  encoder_outputs = [ training_model.get_layer('encoder_lstm_2').output[0],\
                      training_model.get_layer('encoder_lstm_2').output[1],\
                      training_model.get_layer('encoder_lstm_2').output[2]]
  
  
  # Define sampling models
  encoder_model = Model(encoder_inputs, encoder_outputs)
  
  return encoder_model

def create_inference_decoder_model(training_model, idxHour, num_prev_days=dictParams['NumPrevDays']):
  num_lstm_units = dictParams['NumLSTMCells']
  #decoder_inputs = Input(shape=(1,num_features))
  
  decoder_input_name = 'decoder_input_' + str(idxHour)
  decoder_inputs = training_model.get_layer(decoder_input_name).input
  
  decoder_lstm_name = 'decoder_lstm_' + str(idxHour)
  decoder_lstm = training_model.get_layer(decoder_lstm_name)
  
  decoder_output_name = 'decoder_output_' + str(idxHour)
  decoder_dense = training_model.get_layer(decoder_output_name)
  
  num_timesteps_before = dictParams['NumHoursBack']*num_prev_days
  
  decoder_state_input_h = Input(shape=(num_lstm_units,))
  decoder_state_input_c = Input(shape=(num_lstm_units,))
  
  num_features_decoder = len(lAttributesDecoder)


  decoder_encoder_input= Input(shape=(num_timesteps_before,num_lstm_units))

  
  decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
  
  if dictParams['UseConvFilterDec'] == 1:
      filter_layer_name = "dec_filter_layer_" + str(idxHour)
      filter_layer = training_model.get_layer(filter_layer_name)
      filter_layer_out = filter_layer(decoder_inputs)
      num_features = num_features_decoder - dictParams['KernelSizeDec'] + 1
      filter_layer_out = Reshape((1,num_features))(filter_layer_out)
      decoder_lstm_outputs, state_h, state_c = decoder_lstm(
        filter_layer_out, initial_state=decoder_states_inputs)
  else:
      #init the lstm from the training model
      decoder_lstm_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
  
  if dictParams['UseAttention'] == 1:
      attention_layer_name = 'attention_layer_' + str(idxHour)
      attention_layer = training_model.get_layer(attention_layer_name)
      #flat_encoder = Reshape((num_timesteps_before*num_lstm_units,))(decoder_encoder_input)
      #flat_decoder = Reshape((num_lstm_units,))(decoder_lstm_outputs)
      #concat_concat_encoder_decoder = Concatenate(axis=1)([flat_encoder,flat_decoder])  
      attended_decoder = attention_layer(decoder_encoder_input, decoder_lstm_outputs, training=False)  
      decoder_outputs = decoder_dense(attended_decoder)
      #state_c = Multiply()([attended_decoder, state_c])
      decoder_states_outputs = [attended_decoder, state_c]
  else:
      decoder_outputs = decoder_dense(decoder_lstm_outputs)
      decoder_states_outputs = [state_h, state_c]
      
  
  #------------------Model---------------------------------------------------- 
  decoder_model = Model(
    inputs = [decoder_inputs] + decoder_states_inputs + [decoder_encoder_input],   
    outputs = [decoder_outputs] + decoder_states_outputs) 
  
  return decoder_model

       
  
                                         
     
def idxToTargetHour(idx,hour,num_hours_back):
  target_hour = hour - (num_hours_back-idx)
  return target_hour

def get_weights_name(dfDay):
    if dfDay.iloc[0].IsHoliday == 1:
        if dfDay.iloc[0].IsRegularDay == 1:
            return 'reg_holiday_weights.h5'
        else:
            return 'irr_reg_holiday_weights.h5'
    else:
        if dfDay.iloc[0].IsRegularDay == 1:
            return 'reg_workday_weights.h5'
        else:
            return 'irr_reg_workday_weights.h5'
     
def makePredictionWithANN(pathTrainingData, pathEvaluationData,pathResults):
  print(dictParams)

  timeStart = time.time()
  listOfResults=[]
  dfAllData, dfAllDataNormalized, dfAllDataNormalized_Prev, dictFeatures = \
    flt.getDataFrameNormalized(pathTrainingData,pathEvaluationData)

  predictionStart={'Year':2016,'Month':12,'Day':25, 'CurHour':0}#25.12.2016
  predictionEnd={'Year':2017,'Month':12,'Day':31, 'CurHour':23}
  dfPredictionInput = flt.getPredictionInput(predictionStart, predictionEnd, dfAllDataNormalized)#dfAllDataNormalized)
  numEntriesPredInput = dfPredictionInput.shape[0]
  timeEnd = time.time()
  print("Time [s] for import and normalize data: ",timeEnd-timeStart)

  numAllTimeInstances=0
  sumDevAllDays=0

  print("Num entries to predict: ",numEntriesPredInput)

  dfForecastDay=pd.DataFrame()
  dfFilteredTrainingData = pd.DataFrame()
  dfFilteredTrainingDataPrev = pd.DataFrame()
  
  lTargets=[dictParams['ANNTarget']]
  annModel = EncoderDecoderRNN(lAttributesEncoder,lAttributesDecoder,lTargets,0,dictParams['NumHoursBack'],\
                               dictParams['NumHoursAhead'])
  
  reg = None
  ann_model = create_training_model_with_attention(annModel, reg, 'ann_shape.txt')
  ann_model.save_weights('regular_weights.h5')
  ann_model.save_weights('irregular_weights.h5')
  irr_weights_init = False

  decoder_models = []
  encoder_model = create_inference_encoder_model(ann_model)
  
  for idxHour in range(24):
      decoder_models.append(create_inference_decoder_model(ann_model,idxHour))
    
  ##################################################################################
  ##################################################################################
  ##################################################################################

  
  num_collected_days = 0

  #Loop over days
  for idxProcessedEntry in range(0,numEntriesPredInput,24):
    timeStart = time.time()
    startTsSec = dfPredictionInput.iloc[idxProcessedEntry].Timestamp
    endTsSec = startTsSec + 3600*24

    print('Initial num_collected_days', num_collected_days)

    dfForecastDay = dfPredictionInput[(dfPredictionInput.Timestamp>=startTsSec) & \
                                      (dfPredictionInput.Timestamp<endTsSec)].copy()

        

    df_PrevDaysTs = dfAllDataNormalized_Prev[dfAllDataNormalized_Prev['IdxDay'] == dfForecastDay.iloc[0].IdxDay]
        
    isHoliday = dfForecastDay.iloc[0].IsHoliday
    strIsHoliday=""
    if isHoliday==1:
      strIsHoliday="Holiday"
    else:
      strIsHoliday="Workday"
    
    num_last_data = 0

    dfFilteredTrainingData, dfFilteredTrainingDataPrev = \
                 flt.filterDataByDayLengthTempRef(lAttributesFilter, dfForecastDay, dfAllDataNormalized, dfAllDataNormalized_Prev,\
                                                  dictParams['batch_size'], \
                                                  num_last_data, dictParams['NumPrevDays'])

                  

    filter_name='FilterDelta'
    df_sample_weights=pd.DataFrame()
    df_sample_weights[filter_name] = dfFilteredTrainingData[dfFilteredTrainingData.CurHour==0][filter_name]

    #print(df_sample_weights[filter_name].values)

    df_sample_weights['SampleWeights'] =  \
            100*abs(max(df_sample_weights[filter_name])-df_sample_weights[filter_name])/max(df_sample_weights[filter_name])

    print('df_sample_weights.SampleWeights', min(df_sample_weights['SampleWeights']), max(df_sample_weights['SampleWeights']))
    
    l_sample_weights=df_sample_weights['SampleWeights'].values
    l_l_sample_weights = []
    
    for idxHour in range(24):
        l_l_sample_weights.append(l_sample_weights)

    annModel = EncoderDecoderRNN(lAttributesEncoder,lAttributesDecoder,\
                                 lTargets,0,dictParams['NumHoursBack'],\
                                 dictParams['NumHoursAhead'])
            
    InputNodesTraining, OutputNodeTraining = annModel.getInputOutputNodesTraining(dfFilteredTrainingData, \
                                                                                      dfFilteredTrainingDataPrev)
    
    numEpochs = dictParams['NumEpochs']
        
    batch_size_forecast = dictParams['batch_size']
    num_epochs_forecast = numEpochs
        
    if dfForecastDay.iloc[0].IsRegularDay >= 0.49 or irr_weights_init == False:
        ann_model.load_weights('regular_weights.h5')
    else:
        ann_model.load_weights('irregular_weights.h5')

    #------------------FIT-------------------------------------------------------------
    ann_model.fit(InputNodesTraining, OutputNodeTraining, epochs=num_epochs_forecast, \
                  batch_size=batch_size_forecast,verbose=0,\
                  shuffle=False, validation_split = 0.0, callbacks=callbacks_list,\
                  sample_weight=l_l_sample_weights)
        
    print("Model fitted successfully!")
    print("get_slot_names",ann_model.optimizer.get_slot_names())


    if dfForecastDay.iloc[0].IsRegularDay >= 0.49:
        ann_model.save_weights('regular_weights.h5')
    else:
        irr_weights_init = True
        ann_model.save_weights('irregular_weights.h5')

    prevPred = dfForecastDay.iloc[0].PrevLoadPower_Cont #prev, 23
    prevprevPred = dfForecastDay.iloc[0].PrevPrevLoadPower_Cont
    
    #Forecast
    predValue=0
    sumDev=0
        
    timeStartPrediction = time.time()
    dfForecastDayPred = dfForecastDay.copy()     
    #encoder_model = create_inference_encoder_model(ann_model)
    InputNodesEval = annModel.getInputNodesEvaluation(dfForecastDayPred,df_PrevDaysTs)

    print("InputNodesEval", InputNodesEval[0].shape)
    #--------------ENCODER--------------------------------------------------  
    #predict input for the decoder with encoder input 
    encoder_outputs = encoder_model.predict_on_batch(InputNodesEval[0])
    states_value = [encoder_outputs[1], encoder_outputs[2]]
    num_collected_days = num_collected_days - 1

    for idxHour in range(0, dictParams['NumHoursAhead']):
        timeStartSinglePrediction = time.time()
          
        decoder_model = decoder_models[idxHour]
          
        if idxHour > 0:
            prevprevPred = prevPred
            prevPred = predValue
            
        dfForecastDayPred.loc[dfForecastDayPred.CurHour == idxHour,'PrevLoadPower_Cont'] = prevPred
        dfForecastDayPred.loc[dfForecastDayPred.CurHour == idxHour,'PrevPrevLoadPower_Cont'] = prevprevPred
          
        annModel = EncoderDecoderRNN(lAttributesEncoder,lAttributesDecoder,lTargets,idxHour,\
                                    dictParams['NumHoursBack'],\
                                    dictParams['NumHoursAhead']) 
          
        InputNodesEval = annModel.getInputNodesEvaluation(dfForecastDayPred,df_PrevDaysTs)
          
        #------DECODER--------------------------------------
        predValue_array, h, c = decoder_model.predict_on_batch([InputNodesEval[1]] + states_value + [encoder_outputs[0]])
              
        # Update states
        states_value = [h, c]
        #print("predValue", type(predValue))
        predValue = float(predValue_array[0,0])
        #print("predValue", type(predValue))
     
        #tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
        timeEndSinglePrediction = time.time()
          
        featLoadPower = dictFeatures[dictParams['ANNTarget']]
        OutputEval = dfForecastDay.iloc[idxHour].LoadPower_Orig
          
          
        #print("Time single load prediction: ", timeEndSinglePrediction-timeStartSinglePrediction)
        predValueMW = norm.scaleBackInputData(predValue,featLoadPower)
          
        
        dev = (abs(predValueMW-OutputEval)/OutputEval)*100
        sumDev+=dev
          
        result={'Timestamp': int(dfForecastDay.iloc[idxHour].Timestamp),\
                'IdxDay':int(dfForecastDay.iloc[0].IdxDay),\
                'Year': int(dfForecastDay.iloc[0].Year), \
                'Month':int(dfForecastDay.iloc[0].Month), \
                'Day':int(dfForecastDay.iloc[0].Day),\
                'Hour':idxHour,'IsHoliday':dfForecastDay.iloc[0].IsHoliday,\
                'DayOfWeek':int(norm.scaleBackInputData(dfForecastDay.iloc[idxHour].DayOfWeek,dictFeatures['DayOfWeek'])),\
                'IsRegularDay':float(dfForecastDay.iloc[idxHour].IsRegularDay),\
                'Temperature':norm.scaleBackInputData(dfForecastDay.iloc[idxHour].Temperature_Cont,\
                    dictFeatures['Temperature_Cont']),\
                'WetBulb':norm.scaleBackInputData(dfForecastDay.iloc[idxHour].WetBulb_Cont,\
                    dictFeatures['WetBulb_Cont']),\
                'Real':OutputEval,'Predicted':float(predValueMW),'Deviation':dev,\
                'PredictionTime':(timeEndSinglePrediction-timeStartSinglePrediction)}
        listOfResults.append(result)
        
    
    error = sumDev/24

    print("Daily deviation on %s, %4d.%02d.%02d: %4f DailyMinTemp %4f DailyMaxTemp %4f DailyMinTempNextDay %4f Load23 %4f" % (strIsHoliday,dfForecastDay.iloc[0].Year,\
          dfForecastDay.iloc[0].Month,dfForecastDay.iloc[0].Day,sumDev/24,dfForecastDay.iloc[0].DailyMinTemp,\
          dfForecastDay.iloc[0].DailyMaxTemp,dfForecastDay.iloc[0].DailyMinTempNextDay,dfForecastDay.iloc[0].LoadPower_Cont23))
    timeEndPrediction = time.time()
    
    for idxRes in range(len(listOfResults)-24,len(listOfResults)):
      print(listOfResults[idxRes])
    
    timeEnd = time.time()
    print("Total time [s] for ANN model creation and prediction: ",timeEnd-timeStart, " Time for prediction: ", timeEndPrediction-timeStartPrediction)
    
    if dfForecastDay.iloc[0].Year == 2017:
      numAllTimeInstances+=24
      sumDevAllDays+=sumDev
      print("Current MAPE: ",sumDevAllDays/numAllTimeInstances,"%")
        
    
  if numAllTimeInstances > 0:
      print("TotalForecast: ", sumDevAllDays/numAllTimeInstances)
  printListOfResults(pathResults, listOfResults)


  
###################################PREDICTION#########################################################  
pathTrainingData='Data\\' + dictParams['ZoneName'] + '_TrainingData_Python.csv'
pathEvaluationData='Data\\'+ dictParams['ZoneName'] +'_EvaluationData_Python.csv'
pathResults='Data\\Results\\'+ dictParams['ZoneName'] +'_ANN_Results.csv'

print('Make prediction')
makePredictionWithANN(pathTrainingData,pathEvaluationData,pathResults)
print('Print formatted results')
printFormatedResults(pathResults)
