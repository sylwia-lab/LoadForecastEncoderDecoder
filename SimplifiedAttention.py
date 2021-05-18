from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, constraints, initializers, activations
from tensorflow.keras.layers import TimeDistributed, LSTM, Input,  \
                                    Layer, Dense, Dot, Softmax, Concatenate, Reshape, RepeatVector, Multiply
import tensorflow
import math

class SimplifiedAttention(Layer):
  
    def __init__(self, units, timesteps_before, initializer, layer_name, dynamic=True, \
                 cnt_call=0, debug=0):
        super(SimplifiedAttention, self).__init__(name=layer_name)
        self.units = units
        self.timesteps_before = timesteps_before
        self.debug = debug
        self.MatOut = Dense(units, activation='tanh', \
                            kernel_initializer=initializer)
        self.MatOut.trainable = True
        self.cnt_call = cnt_call
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'timesteps_before': self.timesteps_before,
            'debug': self.debug,
            'MatOut': self.MatOut,
            'cnt_call': self.cnt_call,
        })
        return config
    def build(self, input_shape):
        self.output_dim = self.units
        super(SimplifiedAttention, self).build(input_shape)
        #print("build input_shape",input_shape)
    
    def call(self, encoder_outputs, decoder_state, training=True):
        #1 x units => units
        decoder_state_flat = Reshape((self.units,))(decoder_state)
        #timesteps_before x units
        decoder_state_reshaped = RepeatVector(self.timesteps_before)(decoder_state_flat)
        #abs diff
        diff_layer = tensorflow.math.abs(tensorflow.math.subtract(encoder_outputs,decoder_state_reshaped))
        #softmax(diff)
        alpha_layer = tensorflow.nn.softmax(diff_layer)
        #softmax(1-alpha)
        alpha_layer = tensorflow.nn.softmax(1-alpha_layer)
        #alpha x encoder, it's a kind a mask, which deletes all unsimilar entries from encoder_outputs
        sum_multiply_layer = tensorflow.math.multiply(alpha_layer, encoder_outputs)
        #We sum over different states for each time instance t, sum over time dimension
        #target shape: batch x 1 x units
        sum_multiply_layer = tensorflow.reduce_sum(sum_multiply_layer, axis=1)
        #append mask to the last decoded state        
        concatenated_out = tensorflow.concat([sum_multiply_layer, decoder_state_flat],1)
        #target shape: batch x units, corrected previous decoder state
        concatenated_out = self.MatOut(concatenated_out)        
        
        if self.debug == 1 and training == False:
            print('SimplifiedAttention decoder_state', decoder_state.shape)
            print("SimplifiedAttention decoder_state_reshaped",decoder_state_reshaped.shape)
            print('SimplifiedAttention encoder_outputs', encoder_outputs.shape)
            print("SimplifiedAttention alpha_layer", alpha_layer.shape)
            print("SimplifiedAttention sum_multiply_layer", sum_multiply_layer.shape)
            #print("SimplifiedAttention concatenated_out", concatenated_out.shape)
        return concatenated_out
        

    def compute_output_shape(self, input_shape):
        #print("SimplifiedAttention input_shape",input_shape)
        return (input_shape[0], self.output_dim)
    
