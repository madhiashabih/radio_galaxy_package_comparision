import jax
import jax.numpy as jnp
from jax import random, jit, grad
import flax.linen as nn
import optax
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix, f1_score
import time
from util import get_memory_usage, print_metrics
import sys

# https://github.com/dattgoswami/MNIST-Classifier-JAX/blob/main/mnist_jax.py

class NeuralNet(nn.Module):
    num_of_class: int

    @nn.compact
    def __call__(self, x, training=True, dropout_key=None):
        # CNN layers
        x = nn.Conv(features=6, kernel_size=(5, 5), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten the output
        x = x.reshape((x.shape[0], -1))

        # Fully connected layers
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84)(x)
        x = nn.relu(x)

        # Classifier
        x = nn.Dense(features=self.num_of_class)(x)
        return x
        

def main():
    args = sys.argv  # Get the command-line arguments
    run = args[1]
    key = random.PRNGKey(0)

    # Load the MNIST dataset
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))

    # Normalize data
    train_images, train_labels = train_ds['image'], train_ds['label']
    test_images, test_labels = test_ds['image'], test_ds['label']

    train_images = jnp.float32(train_images) / 255.0
    test_images = jnp.float32(test_images) / 255.0

    # Initialize model and optimizer
    model = NeuralNet(num_of_class=10)
    #model = Net()
    rngs = {'params': key, 'dropout': key}
    params = model.init(
        rngs, train_images[0:1])
    tx = optax.adam(0.001)
    opt_state = tx.init(params)

    num_epochs = 5
    batch_size = 32

    # Measure time
    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(train_images), batch_size):
            batch = {
                'image': train_images[i:i+batch_size],
                'label': train_labels[i:i+batch_size],
            }

            # Split key for dropout
            key, dropout_key = random.split(key)

            # Compute loss
            logits = model.apply(params, batch['image'],
                     training=True, dropout_key=dropout_key)

            one_hot = jax.nn.one_hot(batch['label'], 10)
            loss = jnp.mean(optax.softmax_cross_entropy(
                logits=logits, labels=one_hot))

            # Compute gradients
            grads = grad(lambda p, i, l, k: jnp.mean(optax.softmax_cross_entropy(logits=model.apply(
                p, i, training=True, dropout_key=k), labels=jax.nn.one_hot(l, 10))))(params, batch['image'], batch['label'], dropout_key)

            # Update parameters
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if i % (batch_size * 100) == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

    train_time = time.time() - start_time
    
    # Evaluation
    start_time = time.time()
    test_logits = model.apply(params, test_images, training=False)
    test_predictions = jnp.argmax(test_logits, axis=-1)
    inference_time = time.time() - start_time
    
    memory_usage = get_memory_usage()
    
    # Results
    acc = jnp.mean(test_predictions == test_labels)
    test_predictions = jnp.argmax(test_logits, axis=-1)
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    cm = confusion_matrix(test_labels, test_predictions)
    
    print_metrics(acc, f1, cm, train_time, inference_time, memory_usage, "out/jax/mnist/", f"out/jax/mnist/output_{run}.txt")
    
if __name__ == "__main__":
    main()
