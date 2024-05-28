import os
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Function to load shot_features1 and labels
def load_data():
    features = []
    labels = []

    # Load shot_features1 from all files in the directory
    for filename in tqdm(os.listdir('./sfeatures')):
        with open(os.path.join('./sfeatures', filename), 'rb') as f:
            data = pickle.load(f)[1:]

            # Convert data to a numpy array
            data = np.array(data)

            # Append everything except the last column to features
            features.extend(data[:, :-1])

            # Append the last column to labels
            labels.extend(data[:, -1])

    return np.array(features), np.array(labels)


def preprocess_data(features, labels):
    # Convert labels to integers
    labels = labels.astype(int)

    # Remove the 1st, 2nd, 10th, and 11th columns as they are names of the players
    features = np.delete(features, [0, 1, 2, 10, 11], axis=1)

    # Convert features to float32
    features = features.astype(np.float32)

    # Check for NaN values and replace them with 0
    features = np.nan_to_num(features)

    # Normalize the features
    scaler = StandardScaler()

    # Only scale columns - 0, 1, 4, 5, 8, 9, 10
    features[:, [0, 1, 4, 5, 7, 8, 9, 10]] = scaler.fit_transform(features[:, [0, 1, 4, 5, 7, 8, 9, 10]])

    # Save the scaler's mean and scale parameters
    np.save('shot_scaler_mean.npy', scaler.mean_)
    np.save('shot_scaler_scale.npy', scaler.scale_)

    return features, labels


# Load shot_features1 and labels
features, labels = load_data()

# Preprocess the data
features, labels = preprocess_data(features, labels)

# Save 1 instance of a feature vector and label randomly to test the model in the end
random_index = np.random.randint(0, features.shape[0])
random_feature = features[random_index]
random_label = labels[random_index]

# Split the data into training and testing sets based on the labels
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=42)

# Define the custom optimizer with a different learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Define the model architecture with a leaky ReLU activation function
# The model has 4 hidden layers with 512 units each and a dropout layer with a dropout rate of 0.2
# The output layer has 1 unit with a sigmoid activation function
# The sigmoid layer is used to convert the output to a probability
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with the custom optimizer
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Plot the training and validation accuracy
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training and validation loss
plt.plot(model.history.history['loss'], label='Training Loss')
plt.plot(model.history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')

plt.xlabel('Epoch')
plt.legend(loc="upper right")
plt.show()

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(test_features, test_labels)
print("Model Loss:", loss)
print("Model Accuracy:", accuracy)
print(model.summary())

# Print the F1 score
predictions = model.predict(test_features).ravel()
predictions = np.round(predictions)
true_positives = np.sum(predictions * test_labels)
false_positives = np.sum(predictions * (1 - test_labels))
false_negatives = np.sum((1 - predictions) * test_labels)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * precision * recall / (precision + recall)

print("F1 Score:", f1_score)

# Make an ROC curve

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(test_labels, predictions)

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Save the model

model.save('shot_model_v2.keras')