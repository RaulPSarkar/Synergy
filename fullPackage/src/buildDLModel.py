from ast import literal_eval

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, \
	AlphaDropout, Conv1D, Lambda, Concatenate, MaxPooling1D, GlobalMaxPooling1D, Multiply, \
	Conv2D, MaxPooling2D, GlobalAveragePooling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU



from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from tensorflow.keras.models import Model





def buildDL(expr_dim=None, drug_dim=None, expr_hlayers_sizes='[10]', drug_hlayers_sizes='[10]',
                          predictor_hlayers_sizes='[10]', initializer='he_normal', hidden_activation='relu', l1=0,
                          l2=0, input_dropout=0, hidden_dropout=0, optimizer='Adam', learn_rate=0.001):
	"""Build a multi-input deep learning model with separate feature-encoding subnetworks for expression data, drugA
	and drugB, with fully-connected layers in all subnetworks."""
	expr_input = Input(shape=expr_dim, name='expr')
	drug1_input = Input(shape=drug_dim, name='drugA')
	drug2_input = Input(shape=drug_dim, name='drugB')
	expr = dense_submodel(expr_input, hlayers_sizes=expr_hlayers_sizes, l1_regularization=l1, l2_regularization=l2,
	                      hidden_activation=hidden_activation, input_dropout=input_dropout,
	                      hidden_dropout=hidden_dropout)
	drug_submodel = drug_dense_submodel(drug_dim, hlayers_sizes=drug_hlayers_sizes, l1_regularization=l1,
	                                    l2_regularization=l2, hidden_activation=hidden_activation,
	                                    input_dropout=input_dropout, hidden_dropout=hidden_dropout)
	drugA = drug_submodel(drug1_input)
	drugB = drug_submodel(drug2_input)

	concat = concatenate([expr, drugA, drugB])

	# Additional dense layers after concatenating:
	main_branch = dense_submodel(concat, hlayers_sizes=predictor_hlayers_sizes,
	                             l1_regularization=l1, l2_regularization=l2,
	                             hidden_activation=hidden_activation, input_dropout=0,
	                             hidden_dropout=hidden_dropout)
	# Add output layer
	output = Dense(2, activation='linear', kernel_initializer=initializer, name='output')(main_branch)

	# create Model object
	model = Model(inputs=[expr_input, drug1_input, drug2_input], outputs=[output])

	# Define optimizer
	opt_class = dict(getmembers(optimizers))[optimizer]
	if optimizer == 'SGD':
		opt = opt_class(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
	else:
		opt = opt_class(lr=learn_rate)
	
	# Compile model

	model.compile(loss='mean_squared_error', optimizer=opt,
	              metrics=[keras_r2_score, keras_spearman, keras_pearson])

	return model



def drug_dense_submodel(input_dim, hlayers_sizes='[10]', l1_regularization=0, l2_regularization=0,
                        hidden_activation='relu', input_dropout=0, hidden_dropout=0):
	"""Build a dense (fully-connected) submodel for drugs, so that it can be used later on to share weights
	between two drug subnetworks."""
	input_layer = Input(shape=input_dim)

	output = dense_submodel(input_layer, hlayers_sizes, l1_regularization, l2_regularization,
	                        hidden_activation, input_dropout, hidden_dropout)

	submodel = Model(inputs=input_layer, outputs=output, name='drug_dense_submodel')

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
