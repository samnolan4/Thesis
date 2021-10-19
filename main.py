## Original GAN code copied from https://www.tensorflow.org/tutorials/generative/dcgan
## AC-GAN transformation authored by Samuel Nolan

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import time
from IPython import display

# Imports MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Reshapes and normalisee images, converts labels to one-hot
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

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


# Generator Model consists of 3 Transpose Convolutions, using Batch Normalisation and Leaky Relu,
# Takes as input a 110 floats: the first 10 floats are the class (one-hot). The last 100 are random noise
# Generates an image with values between [-1, 1]
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(110,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


generator = make_generator_model()


# Discriminator consists of 3 convolutional layers, including LeakyRelu.
# Outputs 11 values: Value 0: Real or Fake, Values 1-11: One hot class probabilities.
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(11, activation='sigmoid'))

    return model


discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy()


# Discriminator loss is the cross entropy of correctly labelling real or fake + the cross entropy of the correct class.
def discriminator_loss(real_output, fake_output, real_class, fake_class):
    real_loss = cross_entropy(tf.ones_like(real_output[:, 0]), real_output[:, 0])
    fake_loss = cross_entropy(tf.zeros_like(fake_output[:, 0]), fake_output[:, 0])
    class_loss = cross_entropy(real_class[0:real_output.shape[0], :], real_output[:, 1:11]) + cross_entropy(
        fake_class[0:fake_output.shape[0], :], fake_output[:, 1:11])
    total_loss = real_loss + fake_loss + class_loss
    return total_loss


# Generator loss is the cross entropy of 'fooling' the discriminator + the cross entropy of the correct class.
def generator_loss(fake_output, real_class):
    return cross_entropy(tf.ones_like(fake_output[:, 0]), fake_output[:, 0]) + cross_entropy(real_class,
                                                                                             fake_output[:, 1:11])


# Use ADAM optimizer with learning rate 0.0001 for both networks.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Used to save training checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16

# This random noise gets reused when generating images so that we can see a progression
seed1 = tf.random.normal([num_examples_to_generate, noise_dim])

real_classes = tf.Variable(tf.zeros([60000, 10], dtype=tf.float32))


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    # Generates the fake classes for input into the generator.
    fake_classes = tf.one_hot(tf.random.uniform([BATCH_SIZE], minval=0, maxval=9, dtype=tf.int32), depth=10, axis=-1)

    # Generator receives the fake classes + the noise
    input = tf.concat([fake_classes, noise], 1)

    # Retrieves the real classes from the labelled dataset.
    real_classes = images[1]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(input, training=True)

        # Discriminator recieves both a real sample and a fake sample
        real_output = discriminator(images[0], training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output, fake_classes)
        disc_loss = discriminator_loss(real_output, fake_output,
                                       real_classes,
                                       fake_classes)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
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
        generate_and_save_images(generator, discriminator,
                                 epoch + 1,
                                 seed1)

        # Save the model every 10 epochs, also calculates classification accuracy
        if (epoch + 1) % 10 == 0:
            # Calculates the discrimiator accuracy on the test set.
            results = discriminator.predict(test_images)
            metric = tf.keras.metrics.CategoricalAccuracy()
            metric.update_state(test_labels, results[:, 1:11])
            print('Classification Accuracy at Epoch {} is {}'.format(epoch + 1, metric.result().numpy()))

            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, discriminator,
                             epochs,
                             seed1)


def generate_and_save_images(generator, discriminator, epoch, test_input):
    classes = tf.Variable(tf.zeros([16, 10]))

    # Sets classes to be 0-9, 0-5
    for i in range(0, 16):
        classes[i, i % 10].assign(1.0)

    # Combines classes with seeded noise to create input
    test_input = tf.concat([classes, test_input], 1)

    output = generator(test_input, training=False)

    # Plots the output images in a 4x4 matrix.
    fig = plt.figure(figsize=(4, 4))
    for i in range(output.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(output[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


train(train_dataset, EPOCHS)
