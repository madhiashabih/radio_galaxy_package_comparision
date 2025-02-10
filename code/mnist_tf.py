import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
import time
from util import get_memory_usage, print_metrics 
import sys

# Define the model structure similar to JAX
class NeuralNetTensorFlow(models.Model):
    def __init__(self, num_classes=10):
        super(NeuralNetTensorFlow, self).__init__()
        self.conv1 = layers.Conv2D(6, (5, 5), padding='same', activation='relu')
        self.pool1 = layers.AveragePooling2D((2, 2))
        self.conv2 = layers.Conv2D(16, (5, 5), activation='relu')
        self.pool2 = layers.AveragePooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation='relu')
        self.fc2 = layers.Dense(84, activation='relu')
        self.fc3 = layers.Dense(num_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

args = sys.argv  # Get the command-line arguments
run = args[1]

# Load and preprocess MNIST dataset
(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(preprocess).batch(32)
ds_test = ds_test.map(preprocess).batch(32)

# Define model, loss, and optimizer
model = NeuralNetTensorFlow(num_classes=10)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile and train
model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
start_time = time.time()
model.fit(ds_train, epochs=5)
train_time = time.time() - start_time

# Evaluate and print test accuracy
start_time = time.time()
test_loss, test_accuracy = model.evaluate(ds_test)
inference_time = time.time() - start_time

# Get predictions and true labels
y_pred = []
y_true = []

for images, labels in ds_test:
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

# Calculate F1 score and confusion matrix
acc = test_accuracy
f1 = f1_score(y_true, y_pred, average='weighted')
cm = confusion_matrix(y_true, y_pred)
memory_usage = get_memory_usage()

print_metrics(acc, f1, cm, train_time, inference_time, memory_usage, "out/tensorflow/mnist/", f"out/tensorflow/mnist/output_{run}.txt")