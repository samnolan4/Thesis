import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import time
from IPython import display

# Imports MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Reshapes and normalisee images, converts labels to one-hot
train_images = train_images / 255  # Normalize the images to [-1, 1]
train_labels = train_labels.reshape(train_labels.shape[0])
test_images = test_images / 255  # Normalize the images to [-1, 1]
test_labels = test_labels.reshape(test_labels.shape[0])

train_labels = tf.cast(train_labels, dtype=tf.int32)
train_labels = tf.one_hot(train_labels, depth=10, axis=-1)
test_labels = tf.cast(test_labels, dtype=tf.int32)
test_labels = tf.one_hot(test_labels, depth=10, axis=-1)

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Creates a tuple for shuffling and batching
train_dataset = (train_images, train_labels)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# Discriminator consists of 3 convolutional layers, including LeakyRelu.
# Outputs 11 values: Value 0: Real or Fake, Values 1-11: One hot class probabilities.
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[32, 32, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='sigmoid'))

    return model


discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy()

# Discriminator loss is the cross entropy of correctly labelling real or fake + the cross entropy of the correct class.
def discriminator_loss(real_output, real_class):
    class_loss = cross_entropy(real_class[0:real_output.shape[0], :], real_output[:, 0:10])

    return class_loss

# Use ADAM optimizer with learning rate 0.0001 for both networks.
discriminator_optimizer = tf.keras.optimizers.Adam(3e-4)

# Used to save training checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(discriminator_optimizer=discriminator_optimizer,
                                 discriminator=discriminator)

EPOCHS = 200
noise_dim = 100
num_examples_to_generate = 16

real_classes = tf.Variable(tf.zeros([60000, 10], dtype=tf.float32))


@tf.function
def train_step(images):

    # Retrieves the real classes from the labelled dataset.
    real_classes = images[1]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Discriminator recieves both a real sample and a fake sample
        real_output = discriminator(images[0], training=True)

        disc_loss = discriminator_loss(real_output, real_classes)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        i = 0
        for image_batch in dataset:
            # Ignores the final image_batch due to it having a smaller batch size
            if i > (train_images.shape[0] / BATCH_SIZE) - 1:
                break

            train_step(image_batch)
            i = i + 1

        # Generates 16 images at the end of each epoch
        display.clear_output(wait=True)



        # Calculates the discrimiator accuracy on the test set.
        results = discriminator.predict(test_images)
        metric = tf.keras.metrics.CategoricalAccuracy()
        metric.update_state(test_labels, results[:, 0:10])
        print('Classification Accuracy at Epoch {} is {}'.format(epoch + 1, metric.result().numpy()))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)



train(train_dataset, EPOCHS)
