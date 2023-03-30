import base64
import cv2
import joblib
import pathlib
import numpy as np
import os
import keras
from keras import Input
import tensorflow as tf


# Variables
input_size=(224,224)
tokenizer_pkl = r'./static/tokenizer.pkl'
tokenizer=joblib.load(tokenizer_pkl)
max_pad=29
batch_size=100
vocab_size=len(tokenizer.word_index)
embedding_dim=300
dense_dim=512
lstm_units=dense_dim
dropout_rate=0.2
chexnet_weights = r'./static/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
embedding_matrix = np.zeros((vocab_size+1, embedding_dim))

# Functions
def create_chexnet(chexnet_weights = chexnet_weights,input_size = input_size):

  model = tf.keras.applications.DenseNet121(include_top=False,input_shape = input_size+(3,)) #importing densenet the last layer will be a relu activation layer

  #we need to load the weights so setting the architecture of the model as same as the one of the chexnet
  x = model.output #output from chexnet
  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dense(14, activation="sigmoid", name="chexnet_output")(x) #here activation is sigmoid as seen in research paper

  chexnet = tf.keras.Model(inputs = model.input,outputs = x)
  chexnet.load_weights(chexnet_weights)
  chexnet = tf.keras.Model(inputs = model.input,outputs = chexnet.layers[-3].output)  #we will be taking the 3rd last layer (here it is layer before global avgpooling)
  #since we are using attention here
  return chexnet

#This layer will output image backbone features after passing it through chexnet here chexnet will be not be trainable
class Image_encoder(tf.keras.layers.Layer):
  def __init__(self,
               name = "image_encoder_block"
               ):
    super().__init__()
    self.chexnet = create_chexnet()
    self.chexnet.trainable = False
    self.avgpool = tf.keras.layers.AveragePooling2D()
    
  def call(self,data):
    op = self.chexnet(data) 
    op = self.avgpool(op) 
    op = tf.reshape(op,shape = (-1,op.shape[1]*op.shape[2],op.shape[3])) 
    return op 

# Encoder
#Takes image1,image2 and gets the final encoded vector of these
def encoder(image1,image2,dense_dim = dense_dim,dropout_rate = dropout_rate):
  #image1
  im_encoder = Image_encoder()
  bkfeat1 = im_encoder(image1) 
  bk_dense = tf.keras.layers.Dense(dense_dim,name = 'bkdense',activation = 'relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)) 
  bkfeat1 = bk_dense(bkfeat1)

  #image2
  bkfeat2 = im_encoder(image2) 
  bkfeat2 = bk_dense(bkfeat2) 


  #combining image1 and image2
  concat = tf.keras.layers.Concatenate(axis=1)([bkfeat1,bkfeat2]) 
  bn = tf.keras.layers.BatchNormalization(name = "encoder_batch_norm")(concat) 
  dropout = tf.keras.layers.Dropout(dropout_rate,name = "encoder_dropout")(bn)
  return dropout

# luong_global_attention
class luong_global_attention(tf.keras.layers.Layer):

  def __init__(self,dense_dim = dense_dim):
    super().__init__()
    # Intialize variables needed for Concat score function here
    self.W1 = tf.keras.layers.Dense(units = dense_dim) #weight matrix of shape enc_units*dense_dim
    self.W2 = tf.keras.layers.Dense(units = dense_dim) #weight matrix of shape dec_units*dense_dim
    self.V = tf.keras.layers.Dense(units = 1) #weight matrix of shape dense_dim*1 


  def call(self,encoder_output,decoder_h): #here the encoded output will be the concatted image bk features shape
    decoder_h = tf.expand_dims(decoder_h,axis=1) 
    tanh_input = self.W1(encoder_output) + self.W2(decoder_h) 
    tanh_output =  tf.nn.tanh(tanh_input)
    attention_weights = tf.nn.softmax(self.V(tanh_output),axis=1) 
    op = attention_weights*encoder_output #multiply all aplhas with corresponding context vector
    context_vector = tf.reduce_sum(op,axis=1) #summing all context vector over the time period ie input length


    return context_vector,attention_weights

# One_Step_Decoder
class One_Step_Decoder(tf.keras.layers.Layer):
  def __init__(self,vocab_size = vocab_size, embedding_dim = embedding_dim, max_pad = max_pad, dense_dim = dense_dim ,name = "onestepdecoder"):
    # Initialize decoder embedding layer, LSTM and any other objects needed
    super().__init__()
    self.dense_dim = dense_dim
    self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size+1,
                                output_dim = embedding_dim,
                                input_length=max_pad,
                                weights = [embedding_matrix],
                                mask_zero=True, 
                                name = 'onestepdecoder_embedding'
                              )
    self.LSTM = tf.keras.layers.GRU(units=self.dense_dim,
                    #return_sequences=True,
                    return_state=True,
                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=23),
                    recurrent_initializer=tf.keras.initializers.orthogonal(seed=7),
                    name = 'onestepdecoder_LSTM'
                    )

    self.attention = luong_global_attention(dense_dim = dense_dim)
    self.concat = tf.keras.layers.Concatenate(axis=-1)
    self.dense = tf.keras.layers.Dense(dense_dim,name = 'onestepdecoder_embedding_dense',activation = 'relu')
    self.final = tf.keras.layers.Dense(vocab_size+1,activation='softmax')
    self.concat = tf.keras.layers.Concatenate(axis=-1)
    self.add = tf.keras.layers.Add()
  @tf.function
  def call(self,input_to_decoder, encoder_output, decoder_h):#,decoder_c):
    embedding_op = self.embedding(input_to_decoder) #output shape = batch_size*1*embedding_shape (only 1 token)
    

    context_vector,attention_weights = self.attention(encoder_output,decoder_h) #passing hidden state h of decoder and encoder output
    #context_vector shape: batch_size*dense_dim we need to add time dimension
    context_vector_time_axis = tf.expand_dims(context_vector,axis=1)
    #now we will combine attention output context vector with next word input to the lstm here we will be teacher forcing
    concat_input = self.concat([context_vector_time_axis,embedding_op])#output dimension = batch_size*input_length(here it is 1)*(dense_dim+embedding_dim)
    
    output,decoder_h = self.LSTM(concat_input,initial_state = decoder_h)
    #output shape = batch*1*dense_dim and decoder_h,decoder_c has shape = batch*dense_dim
    #we need to remove the time axis from this decoder_output
    

    output = self.final(output)#shape = batch_size*decoder vocab size
    return output,decoder_h,attention_weights

#Decodes the encoder output and caption
class decoder(tf.keras.Model):
  def __init__(self,max_pad = max_pad, embedding_dim = embedding_dim,dense_dim = dense_dim,score_fun='general',batch_size = batch_size,vocab_size = vocab_size):
    super().__init__()
    self.onestepdecoder = One_Step_Decoder(vocab_size = vocab_size, embedding_dim = embedding_dim, max_pad = max_pad, dense_dim = dense_dim)
    self.output_array = tf.TensorArray(tf.float32,size=max_pad)
    self.max_pad = max_pad
    self.batch_size = batch_size
    self.dense_dim =dense_dim
    
  @tf.function
  def call(self,encoder_output,caption):#,decoder_h,decoder_c): #caption : (None,max_pad), encoder_output: (None,dense_dim)
    decoder_h, decoder_c = tf.zeros_like(encoder_output[:,0]), tf.zeros_like(encoder_output[:,0]) #decoder_h, decoder_c
    output_array = tf.TensorArray(tf.float32,size=max_pad)
    for timestep in range(self.max_pad): #iterating through all timesteps ie through max_pad
      output,decoder_h,attention_weights = self.onestepdecoder(caption[:,timestep:timestep+1], encoder_output, decoder_h)
      output_array = output_array.write(timestep,output) #timestep*batch_size*vocab_size

    self.output_array = tf.transpose(output_array.stack(),[1,0,2]) #.stack :Return the values in the TensorArray as a stacked Tensor.)
    return self.output_array

def create_model():
 #inputs to model

  tf.keras.backend.clear_session()
  image1=Input(shape=(input_size+(3,))) #shape = 224,224,3
  image2=Input(shape=(input_size+(3,)))
  caption=Input(shape=(max_pad,))
  encoder_output=encoder(image1,image2,dense_dim,dropout_rate) #shape: (None,28,512)
  output=decoder(max_pad, embedding_dim,dense_dim,batch_size,vocab_size)(encoder_output,caption)
  model=tf.keras.Model(inputs=[image1,image2,caption], outputs=output)
  model_filename = r'./static/Encoder_Decoder_global_attention.h5'
  model_save=model_filename
  model.load_weights(model_save)
  return model,tokenizer

def greedy_search_predict(image1,image2,model, tokenizer):
  """
  Given paths to two x-ray images predicts the impression part of the x-ray in a greedy search algorithm
  """
  # image1 = cv2.imread(image1,cv2.IMREAD_UNCHANGED)/255 
  # image2 = cv2.imread(image2,cv2.IMREAD_UNCHANGED)/255

  image1 = tf.expand_dims(cv2.resize(image1,input_size,interpolation = cv2.INTER_NEAREST),axis=0) #introduce batch and resize
  image2 = tf.expand_dims(cv2.resize(image2,input_size,interpolation = cv2.INTER_NEAREST),axis=0)
  image1 = model.get_layer('image_encoder')(image1)
  image2 = model.get_layer('image_encoder')(image2)
  image1 = model.get_layer('bkdense')(image1)
  image2 = model.get_layer('bkdense')(image2)

  concat = model.get_layer('concatenate')([image1,image2])
  enc_op = model.get_layer('encoder_batch_norm')(concat)  
  enc_op = model.get_layer('encoder_dropout')(enc_op) #this is the output from encoder


  decoder_h,decoder_c = tf.zeros_like(enc_op[:,0]),tf.zeros_like(enc_op[:,0])
  a = []
  pred = []
  for i in range(max_pad):
    if i==0: #if first word
      caption = np.array(tokenizer.texts_to_sequences(['<cls>'])) #shape: (1,1)
    output,decoder_h,attention_weights = model.get_layer('decoder').onestepdecoder(caption,enc_op,decoder_h)#,decoder_c) decoder_c,

    #prediction
    max_prob = tf.argmax(output,axis=-1)  #tf.Tensor of shape = (1,1)
    caption = np.array([max_prob]) #will be sent to onstepdecoder for next iteration
    if max_prob==np.squeeze(tokenizer.texts_to_sequences(['<end>'])): 
      break;
    else:
      a.append(tf.squeeze(max_prob).numpy())
  return tokenizer.sequences_to_texts([a])[0] #here output would be 1,1 so subscripting to open the array

# Prediction Function:

def make_prediction(image_1, image_2):
    '''Driver function to make prediction.'''
    image_1 = f"./{image_1}"
    image_2 = f"./{image_2}"
    model,tokenizer=create_model()
    image1 = cv2.imread(image1,cv2.IMREAD_UNCHANGED)/255 
    image2 = cv2.imread(image2,cv2.IMREAD_UNCHANGED)/255
    predicted_caption=greedy_search_predict(image_1,image_2,model,tokenizer)
    return predicted_caption
    
def ImageCaptionPredict(imgPath):
    '''Driver function to make prediction.'''
    img = data_uri_to_cv2_img(imgPath)
    model,tokenizer=create_model()
    img = img/255
    predicted_caption=greedy_search_predict(img,img,model,tokenizer)
    return predicted_caption

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    im_bytes = base64.b64decode(encoded_data)
    nparr = np.frombuffer(im_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img

# #For prediction
# image_1 = r'./static/uploads/CXR2_IM-0652-2001.png'
# image_2 = r'./static/uploads/CXR3_IM-1384-1001.png'
# model,tokenizer=create_model()
# predicted_caption=greedy_search_predict(image_1,image_2,model,tokenizer)
# print(predicted_caption)
