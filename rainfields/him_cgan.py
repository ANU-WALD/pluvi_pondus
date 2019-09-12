from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import Sequence
import datetime
from data_loader3 import DataLoader
import numpy as np
import os
import prec_verif
import sys
import xarray as xr

class DataGenerator(Sequence):
    def __init__(self, batch_size=4, length=40):
        'Initialization'
        self.batch_size = batch_size
        self.length = length

    def __len__(self):
        return int(self.length/self.batch_size)

    def __getitem__(self, index):
        x = []
        y = []

        a = []
        while len(y) < self.batch_size:
            np.random.seed()
            n = int(np.random.randint(1,6*24*6,size=1)[0])
            a.append(n)
            d = datetime(2018,11,1,0,0) + timedelta(0,10*60*n)
            dp = d - timedelta(0,10*60)
            rf_fp = "/home/lar116/project/pablo/rainfields_data/310_{}_{}.prcp-c10.npy".format(d.strftime("%Y%m%d"), d.strftime("%H%M%S"))
            h8_fp = "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_{}.nc".format(d.strftime("%Y%m%d"))
            h8p_fp = "/home/lar116/project/pablo/rainfields_data/H8_2B_BoM_{}.nc".format(dp.strftime("%Y%m%d"))
            
            if not os.path.exists(rf_fp) or not os.path.exists(h8_fp) or not os.path.exists(h8p_fp):
                continue
           
            h8_ds = xr.open_dataset(h8_fp)
            h8p_ds = xr.open_dataset(h8p_fp)

           
            if np.datetime64(d) not in h8_ds.time.data or np.datetime64(dp) not in h8p_ds.time.data:
                continue
            
            b8 = xr.open_dataset(h8_fp).B8.sel(time=d)[2::2, 402::2].data
            b14 = xr.open_dataset(h8_fp).B14.sel(time=d)[2::2, 402::2].data
            b8p = xr.open_dataset(h8p_fp).B8.sel(time=dp)[2::2, 402::2].data
            b14p = xr.open_dataset(h8p_fp).B14.sel(time=dp)[2::2, 402::2].data
            prec = np.load(rf_fp)[2::2, 402::2]

            x.append(np.stack((b8p,b14p,b8,b14), axis=-1))
            y.append(prec)

        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)[:,:,:,None]

        return y, x

    def on_epoch_end(self):
        pass


def generator_model(img_height=1024, img_width=1024, channels=4, gf=32):
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

    # Image input
    d0 = Input(shape=(img_height, img_width, channels))

    # Downsampling
    d1 = conv2d(d0, gf, strides=2)
    d2 = conv2d(d1, gf*2, strides=2)
    d3 = conv2d(d2, gf*4, strides=2)
    d4 = conv2d(d3, gf*8, strides=2)
    d5 = conv2d(d4, gf*8, strides=2)

    # Upsampling
    u1 = deconv2d(d5, d4, gf*8, strides=2)
    u2 = deconv2d(d4, d3, gf*4, strides=2)
    u3 = deconv2d(u2, d2, gf*2, strides=2)
    u4 = deconv2d(u3, d1, gf, strides=2)
    u5 = UpSampling2D(size=2)(u4)
        
    output_img = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='relu')(u5)

    return Model(inputs=d0, outputs=output_img)


def discriminator_model(img_height=1024, img_width=1024, channels=4, df=32):

    def d_layer(layer_input, filters, strides, f_size=4):
        """Discriminator layer"""
        d = BatchNormalization()(layer_input)
        d = Conv2D(filters, strides=strides, kernel_size=f_size, padding='same')(d)
        return LeakyReLU(alpha=0.2)(d)

    img_A = Input(shape=self.imgA_shape)
    img_B = Input(shape=self.imgB_shape)

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, self.df, 2)
    d2 = d_layer(d1, df*2, 2)
    d3 = d_layer(d2, df*4, 2)
    d4 = d_layer(d3, df*8, 2)

    validity = Conv2D(1, strides=1, kernel_size=4, padding='same')(d4)
        
    return Model([img_A, img_B], validity)

"""
def sample_images(epoch, batch_i):
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
                axs[i,j].imshow(gen_imgs[cnt])

            axs[i,j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
    plt.close()
"""


# Configure data loader
data_loader = DataGenerator(batch_size=4, length=40)

# Build and compile the discriminator
discriminator = discriminator_model()
print(discriminator.summary())
optimizer = Adam(0.0002, 0.5, decay=0.0002/400)
discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

#-------------------------
# Construct Computational
#   Graph of Generator
#-------------------------

# Build the generator
generator = generator_model()
print(generator.summary())

# Input images and their conditioning images
img_A = Input(shape=(1024,1024,1))
img_B = Input(shape=(1024,1024,4))
print(img_A.shape, img_B.shape)
print("_______________")

# By conditioning on B generate a fake version of A
fake_A = generator(img_B)
print(fake_A.shape)
print("_______________")

# For the combined model we will only train the generator
#self.discriminator.trainable = False

# Discriminators determines validity of translated images / condition pairs
valid = discriminator([fake_A, img_B])

combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimizer)



epochs=400 
batch_size=10 
sample_interval=200
# Calculate output shape of D (PatchGAN)
disc_patch = (10, 15, 1)

start_time = datetime.datetime.now()

# Adversarial loss ground truths
valid = np.ones((batch_size,) + self.disc_patch)
fake = np.zeros((batch_size,) + self.disc_patch)

for epoch in range(epochs):
    for batch_i, (imgs_A, imgs_B) in enumerate(data_loader):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Condition on B and generate a translated version
        fake_A = generator.predict(imgs_B)

        # Train the discriminators (original images = real / generated = Fake)
        d_loss_real = discriminator.train_on_batch([imgs_A, imgs_B], valid)
        d_loss_fake = discriminator.train_on_batch([fake_A, imgs_B], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -----------------
        #  Train Generator
        # -----------------

        # Train the generators
        g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time))

        # If at save interval => save generated image samples
        #if batch_i % sample_interval == 0:
        #sample_images(epoch, batch_i)
        #self.verif_metrics(epoch, epochs, batch_i)
