import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import time
from IPython import display

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
#train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
train_images = train_images / 255  # Normalize the images to [-1, 1]
train_labels = train_labels.reshape(train_labels.shape[0])
#test_images = test_images.reshape(test_images.shape[0], 32, 32, 3).astype('float32')
test_images = test_images / 255  # Normalize the images to [-1, 1]
test_labels = test_labels.reshape(test_labels.shape[0])

train_labels = tf.cast(train_labels, dtype=tf.int32)
train_labels = tf.one_hot(train_labels, depth=10, axis=-1)
test_labels = tf.cast(test_labels, dtype=tf.int32)
test_labels = tf.one_hot(test_labels, depth=10, axis=-1)

BUFFER_SIZE = 59904
BATCH_SIZE = 256

train_dataset = (train_images, train_labels)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(110,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 32, 32, 3)

    return model

generator = make_generator_model()

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[32, 32, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(11, activation='sigmoid'))

    return model

discriminator = make_discriminator_model()
#classifier = discriminator[1:11]

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output, real_class, fake_class):
    real_loss = cross_entropy(tf.ones_like(real_output[:,0]), real_output[:,0])
    fake_loss = cross_entropy(tf.zeros_like(fake_output[:,0]), fake_output[:,0])
    class_loss = cross_entropy(real_class[0:real_output.shape[0], :], real_output[:, 1:11]) + cross_entropy(fake_class[0:fake_output.shape[0], :], fake_output[:, 1:11])
    total_loss = real_loss + fake_loss + class_loss
    return total_loss

def generator_loss(fake_output, real_class):
    return cross_entropy(tf.ones_like(fake_output[:,0]), fake_output[:,0]) + cross_entropy(real_class, fake_output[:,1:11])


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50000
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed1 = tf.random.normal([num_examples_to_generate, noise_dim])
real_classes = tf.Variable(tf.zeros([60000, 10], dtype=tf.float32))

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, batch_num):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    fake_classes = tf.one_hot(tf.random.uniform([BATCH_SIZE], minval=0, maxval=9, dtype=tf.int32), depth=10, axis=-1)
    real_classes = images[1]

    input = tf.concat([fake_classes, noise], 1)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(input, training=True)

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
  k = 0
  for epoch in range(epochs):
    start = time.time()
    i = 0
    for image_batch in dataset:
        if i > (train_images.shape[0]/BATCH_SIZE) - 1:
            break
        train_step(image_batch, i)
        i = i+1
    display.clear_output(wait=True)

    # Save the model every 15 epochs
    if (epoch) % 500 == 0:

      results = discriminator.predict(test_images)
      metric = tf.keras.metrics.CategoricalAccuracy()
      metric.update_state(test_labels, results[:, 1:11])
      generate_and_save_images(generator, discriminator,
                               epoch + 1,
                               seed1)
      print(metric.result().numpy())
      #checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  generate_and_save_images(generator, discriminator,
                           epochs,
                           seed1)

def generate_and_save_images(model, discriminator, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  test = tf.Variable(tf.zeros([16, 10]))
  for i in range(0, 16):
      test[i, i%10].assign(1.0)
  test_input = tf.concat([test,test_input], 1)
  predictions = model(test_input, training=False)
  classification = discriminator(predictions, training=False)
  print(classification[:,:])
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i])
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

train(train_dataset, EPOCHS)