import numpy as np
import sys
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
sys.path.append("..")
from src.buildDLModel import buildDL




model = buildDL(exp, drugdim, 10, 10, 10)

history = best_model.fit(x=train_dataset, epochs=settings['epochs'], batch_size=settings['batch_size'],
                            callbacks=[EarlyStopping(patience=15, restore_best_weights=True),
                                    CSVLogger(os.path.join(output_dir, 'training.log'))],
                            validation_data=(val_dataset.X_dict, val_dataset.y), workers=6,
                            use_multiprocessing=False, validation_batch_size=64)#, class_weight=weights)


# best number of epochs
best_n_epochs = np.argmin(history.history['val_loss']) + 1
print('best n_epochs: %s' % best_n_epochs)
