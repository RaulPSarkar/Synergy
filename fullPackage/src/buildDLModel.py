from ast import literal_eval

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, \
	AlphaDropout, Conv1D, Lambda, Concatenate, MaxPooling1D, GlobalMaxPooling1D, Multiply, \
	Conv2D, MaxPooling2D, GlobalAveragePooling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D,Flatten,Dropout,Dense
from tensorflow.keras  import optimizers


from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from tensorflow.keras.models import Model

import sys
sys.path.append("..")









def buildDL( expr_dim=None, drug_dim=None, coeffs_dim=None, useCoeffs=False, useDrugs=True, useSingleAgent=False, mixedModel=False, expr_hlayers_sizes='[10]', drug_hlayers_sizes='[10]', coeffs_hlayers_sizes='[10]',
                          predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0,
                          l2=0, input_dropout=0, hidden_dropout=0, learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA
	and drugB, with fully-connected layers in all subnetworks."""
	
	if(mixedModel):
		useCoeffs=False


	expr_input = Input(shape=expr_dim, name='expr')
	if(useCoeffs):
		coeffs1_input = Input(shape=coeffs_dim, name='coeffsA')
		coeffs2_input = Input(shape=coeffs_dim, name='coeffsB')
	if(useDrugs):
		drug1_input = Input(shape=drug_dim, name='drugA')
		drug2_input = Input(shape=drug_dim, name='drugB')
	if(useSingleAgent):
		singleAgentInput = Input(shape=4, name='singleAgent')


	expr = dense_submodel(expr_input, hlayers_sizes=expr_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                      hidden_activation=hidden_activation, input_dropout=input_dropout,
	                      hidden_dropout=hidden_dropout)
	
	

	if(useDrugs):
		drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
											l2_regularization=l2, hidden_activation=hidden_activation,
											input_dropout=input_dropout, hidden_dropout=hidden_dropout)
		
		drugA = drug_submodel(drug1_input)
		drugB = drug_submodel(drug2_input)



	if(useCoeffs):
		coeffs_submodel = drug_dense_submodel(coeffs_dim, hlayers_sizes=coeffs_hlayers_sizes, l1_regularization=l1,
										l2_regularization=l2, hidden_activation=hidden_activation,
										input_dropout=input_dropout, hidden_dropout=hidden_dropout, layerName='coeffs_dense_submodel')

		coeffsA = coeffs_submodel(coeffs1_input)
		coeffsB = coeffs_submodel(coeffs2_input)

	fullConcat = [expr]
	if(useCoeffs):
		fullConcat.append(coeffsA)
		fullConcat.append(coeffsB)
	if(useDrugs):
		fullConcat.append(drugA)
		fullConcat.append(drugB)
	if(useSingleAgent):
		fullConcat.append(singleAgentInput)



	concat = concatenate(fullConcat)


	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	
	# Add output layer
	output = Dense(2, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	
	fullInputs = [expr_input]
	if(useCoeffs):
		fullInputs.append(coeffs1_input)
		fullInputs.append(coeffs2_input)
	if(useDrugs):
		fullInputs.append(drug1_input)
		fullInputs.append(drug2_input)
	if(useSingleAgent):
		fullInputs.append(singleAgentInput)



	model = Model(inputs=fullInputs, outputs=[output])
	#model = Model(inputs=cnnInp, outputs=[output])

	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate))


	return model



def drug_dense_submodel(input_dim, hlayers_sizes='[10]', l1_regularization=0, l2_regularization=0,
                        hidden_activation='relu', input_dropout=0, hidden_dropout=0, layerName='drug_dense_submodel'):
	"""Build a dense (fully-connected) submodel for drugs, so that it can be used later on to share weights
	between two drug subnetworks."""
	input_layer = Input(shape=input_dim)

	output = dense_submodel(input_layer, hlayers_sizes, l1_regularization, l2_regularization,
	                        hidden_activation, input_dropout, hidden_dropout)

	submodel = Model(inputs=input_layer, outputs=output, name=layerName)

	return submodel






def dense_submodel(input_layer, hlayers_sizes='[10]', l1_regularization=0, l2_regularization=0,
                   hidden_activation='relu', input_dropout=0, hidden_dropout=0):
	"""Build a dense (fully-connected) subnetwork."""
	hlayers_sizes = literal_eval(hlayers_sizes)  # because hlayers_sizes was passed as a string

	if hidden_activation == 'selu':
		# selu must be used with the LecunNormal initializer and AlphaDropout instead of normal Dropout
		initializer = 'lecun_normal'
		dropout = AlphaDropout
		batchnorm = False
	else:
		initializer = 'he_normal'
		dropout = Dropout
		batchnorm = True

	if input_dropout > 0:
		x = dropout(rate=input_dropout)(input_layer)
	else:
		x = input_layer

	for i in range(len(hlayers_sizes)):
		x = Dense(units=hlayers_sizes[i], kernel_initializer=initializer,
		          kernel_regularizer=l1_l2(l1=l1_regularization, l2=l2_regularization))(x)
		if hidden_activation.lower() == 'leakyrelu':
			x = LeakyReLU()(x)
		elif hidden_activation.lower() == 'prelu':
			x = PReLU()(x)
		else:
			x = Activation(hidden_activation)(x)

		if batchnorm:
			x = BatchNormalization()(x)

		if hidden_dropout > 0:
			x = dropout(rate=hidden_dropout)(x)

	return x




def cnnOmicsModel(useDrugs=False, useCoeffs=False, omicsOutputNeurons = 64, drugDim=1024, coeffsDim=0, sizeOmics = 0, filters=32, secondFilter=32, kernelSize=3, secondKernel=3, coeffsHlayerSizes= '[10]',drugHlayerSizes= '[10]',predictorHlayersSizes='[10]', cnnActivation = 'relu', hiddenActivation='relu', initializer='he_normal', hiddenDropout=0, l1=0, l2=0, useSingleAgent = True, learnRate = 0.001):


	if(useDrugs):
		drug1_input = Input(shape=drugDim, name='drugA')
		drug2_input = Input(shape=drugDim, name='drugB')

	if(useCoeffs):
		coeffs1_input = Input(shape=coeffsDim, name='coeffsA')
		coeffs2_input = Input(shape=coeffsDim, name='coeffsB')



	inputOmicsLayer = Input(shape=(3, sizeOmics))
	y = tf.keras.layers.Conv1D(filters, kernelSize, activation='relu', input_shape=(3, sizeOmics), padding='same')(inputOmicsLayer)
	yy = tf.keras.layers.Conv1D(secondFilter, secondKernel, activation='relu', input_shape=(3, sizeOmics), padding='same')(y)
	z = tf.keras.layers.Flatten()(yy)
	w = tf.keras.layers.Dense(omicsOutputNeurons)(z)


	
		
		

	fullInputs = [inputOmicsLayer]
	inputsForMain = [w]
	if(useSingleAgent):
		singleAgentInput = Input(shape=4, name='singleAgent')
		inputsForMain.append(singleAgentInput)
		fullInputs.append(singleAgentInput)
	if(useDrugs):
		fullInputs.append(drug1_input)
		fullInputs.append(drug2_input)

		drug_submodel = drug_dense_submodel(drugDim, hlayers_sizes=drugHlayerSizes, l1_regularization=l1,
							l2_regularization=l2, hidden_activation=hiddenActivation,
							input_dropout=0, hidden_dropout=hiddenDropout)
		drugA = drug_submodel(drug1_input)
		drugB = drug_submodel(drug2_input)

		inputsForMain.append(drugA)
		inputsForMain.append(drugB)

	if(useCoeffs):
		fullInputs.append(coeffs1_input)
		fullInputs.append(coeffs2_input)
		coeffs_submodel = drug_dense_submodel(coeffsDim, hlayers_sizes=coeffsHlayerSizes, l1_regularization=l1,
										l2_regularization=l2, hidden_activation=hiddenActivation,
										input_dropout=0, hidden_dropout=hiddenDropout, layerName='coeffs_dense_submodel')

		coeffsA = coeffs_submodel(coeffs1_input)
		coeffsB = coeffs_submodel(coeffs2_input)
		inputsForMain.append(coeffsA)
		inputsForMain.append(coeffsB)



	inputsForMain = concatenate(inputsForMain)

	# Additional dense layers after concatenating:
	mainBranch = dense_submodel(inputsForMain, hlayers_sizes=predictorHlayersSizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hiddenActivation, input_dropout=0,
	                             hidden_dropout=hiddenDropout)
	
	# Add output layer

	output = Dense(2, activation='linear', kernel_initializer=initializer, name='output')(mainBranch)

	model = Model(inputs=fullInputs , outputs=[output], name='CNN')
	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate))
	#return model

	return model


