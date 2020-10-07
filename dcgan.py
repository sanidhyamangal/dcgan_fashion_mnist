"""
DCGAN
author:Sanidhya Mangal
github:sanidhyamangal
"""

import os  # for os related ops
import time  # for time related ops

import matplotlib.pyplot as plt  # for plotting
import PIL  # for image related ops
import tensorflow as tf  # for deep learning related steps

from data_handler import data_loader_csv_unsupervisied # data loader for the DCGAN using helper library

# Reterieve the data for the training
#!wget https://gitlab.com/sanidhyamangal/datasets/-/raw/master/fashion-mnist_train.csv

# loading the train dataset

train_dataset = data_loader_csv_unsupervisied("./fashion-mnist_train.csv", 256)

# create a sample model for the dcgan Generation
class DCGANGenerative(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(DCGANGenerative, self).__init__(*args, **kwargs)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256,
                                  use_bias=False,
                                  input_shape=(100, )),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            # reshape function
            tf.keras.layers.Reshape((7, 7, 256)),

            # first conv2d transpose
            tf.keras.layers.Conv2DTranspose(filters=128,
                                            kernel_size=(5, 5),
                                            strides=(1, 1),
                                            padding="same",
                                            use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            # second conv2d transpose
            tf.keras.layers.Conv2DTranspose(filters=64,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding="same",
                                            use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            # third conv2d transpose
            tf.keras.layers.Conv2DTranspose(filters=1,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding="same",
                                            use_bias=False,
                                            activation=tf.nn.tanh)
        ])

    def call(self, inputs):
        return self.model(inputs)

# create a sample model for the dcgan discrimination
class DCGANDiscriminative(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(DCGANDiscriminative, self).__init__(*args, **kwargs)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,
                                   kernel_size=(5, 5),
                                   strides=(2, 2),
                                   padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            # first conv2d transpose
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(5, 5),
                                   strides=(1, 1),
                                   padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            # third conv2d transpose
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)

# create a generator
generator = DCGANGenerative()

# create a discriminator
discriminator = DCGANDiscriminative()

# create a loss for models
cross_entropy = tf.losses.BinaryCrossentropy(from_logits=True)


# create a loss for generative
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# create a loss function for discriminator
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# create an optimizer for both generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# create a checkpoint for the models
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# META data for training purpose
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
BATCH_SIZE = 256

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# train step function
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


# function for performing training on multiple batches
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           seed)

# call train function
train(train_dataset, EPOCHS)