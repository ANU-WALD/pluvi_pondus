from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
from data_loader3 import DataLoader
import numpy as np
import os
import prec_verif
import sys

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 240
        self.img_cols = 360
        self.channels = 3
        self.imgA_shape = (self.img_rows, self.img_cols, 1)
        self.imgB_shape = (self.img_rows, self.img_cols, 3)

        # Configure data loader
        self.dataset_name = 'gan_era5'
        self.data_loader = DataLoader()

        # Calculate output shape of D (PatchGAN)
        self.disc_patch = (10, 15, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5, decay=0.0002/400)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        print(self.discriminator.summary())

        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()
        print(self.generator.summary())

        # Input images and their conditioning images
        img_A = Input(shape=self.imgA_shape)
        img_B = Input(shape=self.imgB_shape)
        print(img_A.shape, img_B.shape)
        print("_______________")

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)
        print(fake_A.shape)
        print("_______________")

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)


    def build_generator(self):
        """U-Net Generator"""
    
        def conv2d(layer_input, filters, strides, f_size=4):
            """Layers used during downsampling"""
            e = BatchNormalization()(layer_input)
            return Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', activation='relu')(e)

        def deconv2d(layer_input, skip_input, filters, strides, f_size=4):
            """Layers used during upsampling"""
            d = BatchNormalization()(layer_input)
            d = UpSampling2D(size=strides)(d)
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(d)
            d = Concatenate()([d, skip_input])
            return d

        n_filters = 32
        # Image input
        d0 = Input(shape=(240,360,3))

        # Downsampling
        d1 = conv2d(d0, n_filters, strides=2)
        d2 = conv2d(d1, n_filters*2, strides=3)
        d3 = conv2d(d2, n_filters*4, strides=2)
        d4 = conv2d(d3, n_filters*8, strides=2)
        #d5 = conv2d(d4, n_filters*8, strides=2)

        # Upsampling
        #u1 = deconv2d(d5, d4, n_filters*8, strides=2)
        u2 = deconv2d(d4, d3, n_filters*4, strides=2)
        u3 = deconv2d(u2, d2, n_filters*2, strides=2)
        u4 = deconv2d(u3, d1, n_filters, strides=3)

        u5 = UpSampling2D(size=2)(u4)
        output_img = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='relu')(u5)

        return Model(inputs=d0, outputs=output_img)


    def build_discriminator(self):

        def d_layer(layer_input, filters, strides, f_size=4, bn=True):
            """Discriminator layer"""
            d = BatchNormalization()(layer_input)
            d = Conv2D(filters, strides=strides, kernel_size=f_size, padding='same')(d)
            return LeakyReLU(alpha=0.2)(d)

        img_A = Input(shape=self.imgA_shape)
        img_B = Input(shape=self.imgB_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, 2)
        d2 = d_layer(d1, self.df*2, 3)
        d3 = d_layer(d2, self.df*4, 2)
        d4 = d_layer(d3, self.df*8, 2)

        validity = Conv2D(1, strides=1, kernel_size=4, padding='same')(d4)
        
        return Model([img_A, img_B], validity)


    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
            print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time))

                # If at save interval => save generated image samples
                #if batch_i % sample_interval == 0:
            self.sample_images(epoch, batch_i)
            #self.verif_metrics(epoch, epochs, batch_i)


    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3
        
        imgs_A, imgs_B = self.data_loader.load_data(batch_size=200, is_testing=True)
        fake_A = self.generator.predict(imgs_B)
        print("Test MSE: {} MAE: {}".format(np.mean(np.square(imgs_A - fake_A)), np.mean(np.abs(imgs_A - fake_A))))

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B[:,:,:,0], fake_A[:,:,:,0], imgs_A[:,:,:,0]])


        titles = ['Input', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if i == 0:
                    axs[i,j].imshow(gen_imgs[cnt])
                else:
                    axs[i,j].imshow(np.log(1+gen_imgs[cnt]), vmin=0, vmax=np.log(21), cmap=raincmp)

                axs[i,j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=400, batch_size=10, sample_interval=200)
