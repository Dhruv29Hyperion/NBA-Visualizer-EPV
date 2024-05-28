import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Activation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

features = []
labels = []

# Load data from all files in the directory
for filename in tqdm(os.listdir('./zones')):
    with open(os.path.join('./zones', filename), 'rb') as f:
        data = pickle.load(f)[1:]
        data = np.array(data)
        features.extend(data[:, :-3])
        labels.extend(data[:, -3:])

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Convert features to float32
features = features.astype(np.float32)

# Normalize the last column of the features
scaler = StandardScaler()
features[:, -1] = scaler.fit_transform(features[:, -1].reshape(-1, 1)).reshape(-1)

# Save the scaler's mean and scale parameters
np.save('zone_scaler_mean.npy', scaler.mean_)
np.save('zone_scaler_scale.npy', scaler.scale_)

# Encode categorical variables
encoder_shot_zone = LabelEncoder()
labels_shot_zone = to_categorical(encoder_shot_zone.fit_transform(labels[:, 0]))

encoder_shot_area = LabelEncoder()
labels_shot_area = to_categorical(encoder_shot_area.fit_transform(labels[:, 1]))

encoder_shot_range = LabelEncoder()
labels_shot_range = to_categorical(encoder_shot_range.fit_transform(labels[:, 2]))

# Save the encoders
pickle.dump(encoder_shot_zone, open('encoder_shot_zone.pkl', 'wb'))
pickle.dump(encoder_shot_area, open('encoder_shot_area.pkl', 'wb'))
pickle.dump(encoder_shot_range, open('encoder_shot_range.pkl', 'wb'))

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features, np.column_stack(
    (labels_shot_zone, labels_shot_area, labels_shot_range)), test_size=0.2, random_state=42)

# Neural Network Architecture
input_layer = Input(shape=(X_train.shape[1],))
x = Dense(128)(input_layer)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(64)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

# Output layers: assuming 3 outputs each with its own softmax activation
output_shot_zone = Dense(labels_shot_zone.shape[1], activation='softmax', name='shot_zone')(x)
output_shot_area = Dense(labels_shot_area.shape[1], activation='softmax', name='shot_area')(x)
output_shot_range = Dense(labels_shot_range.shape[1], activation='softmax', name='shot_range')(x)

# Build and compile the model
model = Model(inputs=input_layer, outputs=[output_shot_zone, output_shot_area, output_shot_range])

# Set the learning rate to a specific value
optimizer = Adam(learning_rate=0.001)  # You can adjust this value as needed

model.compile(optimizer=optimizer,
              loss={'shot_zone': 'categorical_crossentropy',
                    'shot_area': 'categorical_crossentropy',
                    'shot_range': 'categorical_crossentropy'},
              metrics={'shot_zone': ['accuracy'],
                       'shot_area': ['accuracy'],
                       'shot_range': ['accuracy']})

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001, verbose=1)

# Train the model
model.fit(X_train, {'shot_zone': Y_train[:, :labels_shot_zone.shape[1]],
                    'shot_area': Y_train[:,
                                 labels_shot_zone.shape[1]:labels_shot_zone.shape[1] + labels_shot_area.shape[1]],
                    'shot_range': Y_train[:, -labels_shot_range.shape[1]:]},
          epochs=10, batch_size=32, validation_split=0.2, callbacks=[reduce_lr])

# Evaluate the model
results = model.evaluate(X_test, [Y_test[:, :labels_shot_zone.shape[1]],
                                  Y_test[:,
                                  labels_shot_zone.shape[1]:labels_shot_zone.shape[1] + labels_shot_area.shape[1]],
                                  Y_test[:, -labels_shot_range.shape[1]:]])

print(results)

# Save the model
model.save('zone_model_v2.keras')




