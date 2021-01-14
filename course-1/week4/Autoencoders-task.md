# Denoising Autoencoders And Where To Find Them

Today we're going to train deep autoencoders and apply them to faces and similar images search.

Our new test subjects are human faces from the [lfw dataset](http://vis-www.cs.umass.edu/lfw/).

# Import stuff


```python
import sys
sys.path.append("..")
import grading
```


```python
import tensorflow as tf
import tensorflow.keras.layers as L, tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from lfw_dataset import load_lfw_dataset
%matplotlib inline
import matplotlib.pyplot as plt
import download_utils
import keras_utils
import numpy as np
```

# Load dataset
Dataset was downloaded for you. Relevant links (just in case):
- http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
- http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
- http://vis-www.cs.umass.edu/lfw/lfw.tgz


```python
# load images
X, attr = load_lfw_dataset(use_raw = True, dimx = 32, dimy = 32)
IMG_SHAPE = X.shape[1:]

# center images
X = X.astype('float32') / 255.0 - 0.5

# split
X_train, X_test = train_test_split(X, test_size = 0.1, random_state = 42)
```


      0%|          | 0/18983 [00:00<?, ?it/s]



```python
def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
```


```python
plt.title('sample images')

for i in range(6):
    plt.subplot(2,3,i + 1)
    show_image(X[i])

print("X shape:", X.shape)
print("attr shape:", attr.shape)

# try to free memory
del X
import gc
gc.collect()
```

    X shape: (13143, 32, 32, 3)
    attr shape: (13143, 73)





    15




    
![png](images/autoencoder_output_7_2.png)
    


# Autoencoder architecture

Let's design autoencoder as two sequential keras models: the encoder and decoder respectively.

We will then use symbolic API to apply and train these models.

<img src="images/autoencoder.png" style="width:50%">

# First step: PCA

Principial Component Analysis is a popular dimensionality reduction method. 

Under the hood, PCA attempts to decompose object-feature matrix $X$ into two smaller matrices: $W$ and $\hat W$ minimizing _mean squared error_:

$$\|(X W) \hat{W} - X\|^2_2 \to_{W, \hat{W}} \min$$
- $X \in \mathbb{R}^{n \times m}$ - object matrix (**centered**);
- $W \in \mathbb{R}^{m \times d}$ - matrix of direct transformation;
- $\hat{W} \in \mathbb{R}^{d \times m}$ - matrix of reverse transformation;
- $n$ samples, $m$ original dimensions and $d$ target dimensions;

In geometric terms, we want to find d axes along which most of variance occurs. The "natural" axes, if you wish.

<img src="images/pca.png" style="width:30%">


PCA can also be seen as a special case of an autoencoder.

* __Encoder__: X -> Dense(d units) -> code
* __Decoder__: code -> Dense(m units) -> X

Where Dense is a fully-connected layer with linear activaton:   $f(X) = W \cdot X + \vec b $


Note: the bias term in those layers is responsible for "centering" the matrix i.e. substracting mean.


```python
def build_pca_autoencoder(img_shape, code_size):
    """
    Here we define a simple linear autoencoder as described above.
    We also flatten and un-flatten data to be compatible with image shapes
    """
    
    encoder = tf.keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Flatten())                  #flatten image to vector
    encoder.add(L.Dense(code_size))           #actual encoder

    decoder = tf.keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(np.prod(img_shape)))  #actual decoder, height*width*3 units
    decoder.add(L.Reshape(img_shape))         #un-flatten
    
    return encoder, decoder
```

Meld them together into one model:


```python
encoder, decoder = build_pca_autoencoder(IMG_SHAPE, code_size = 32)

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = tf.keras.Model(inputs = inp, outputs = reconstruction)
autoencoder.compile(optimizer = 'adamax', loss = 'mse')

autoencoder.fit (
    x = X_train, y = X_train, epochs = 15,
    validation_data = [X_test, X_test],
    # callbacks = [keras_utils.TqdmProgressCallback()],
    verbose = 0
)
```




    <tensorflow.python.keras.callbacks.History at 0x7f623d97ce10>




```python
def visualize(img, encoder, decoder):
    """Draws original, encoded and decoded images"""
    code = encoder.predict(img[None])[0]  # img[None] is the same as img[np.newaxis, :]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1] // 2, -1]))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()
```


```python
score = autoencoder.evaluate(X_test, X_test, verbose = 0)
print("PCA MSE:", score)

for i in range(5):
    img = X_test[i]
    visualize(img, encoder, decoder)
```

    PCA MSE: 0.006544346455484629



    
![png](images/autoencoder_output_14_1.png)
    



    
![png](images/autoencoder_output_14_2.png)
    



    
![png](images/autoencoder_output_14_3.png)
    



    
![png](images/autoencoder_output_14_4.png)
    



    
![png](images/autoencoder_output_14_5.png)
    


# Going deeper: convolutional autoencoder

PCA is neat but surely we can do better. This time we want you to build a deep convolutional autoencoder by... stacking more layers.

## Encoder

The **encoder** part is pretty standard, we stack convolutional and pooling layers and finish with a dense layer to get the representation of desirable size (`code_size`).

We recommend to use `activation='elu'` for all convolutional and dense layers.

We recommend to repeat (conv, pool) 4 times with kernel size (3, 3), `padding='same'` and the following numbers of output channels: `32, 64, 128, 256`.

Remember to flatten (`L.Flatten()`) output before adding the last dense layer!

## Decoder

For **decoder** we will use so-called "transpose convolution". 

Traditional convolutional layer takes a patch of an image and produces a number (patch -> number). In "transpose convolution" we want to take a number and produce a patch of an image (number -> patch). We need this layer to "undo" convolutions in encoder. We had a glimpse of it during week 3 (watch [this video](https://www.coursera.org/learn/intro-to-deep-learning/lecture/auRqf/a-glimpse-of-other-computer-vision-tasks) starting at 5:41).

Here's how "transpose convolution" works:
<img src="images/transpose_conv.jpg" style="width:60%">
In this example we use a stride of 2 to produce 4x4 output, this way we "undo" pooling as well. Another way to think about it: we "undo" convolution with stride 2 (which is similar to conv + pool).

You can add "transpose convolution" layer in Keras like this:
```python
L.Conv2DTranspose(filters=?, kernel_size=(3, 3), strides=2, activation='elu', padding='same')
```

Our decoder starts with a dense layer to "undo" the last layer of encoder. Remember to reshape its output to "undo" `L.Flatten()` in encoder.

Now we're ready to undo (conv, pool) pairs. For this we need to stack 4 `L.Conv2DTranspose` layers with the following numbers of output channels: `128, 64, 32, 3`. Each of these layers will learn to "undo" (conv, pool) pair in encoder. For the last `L.Conv2DTranspose` layer use `activation=None` because that is our final image.


```python
# Let's play around with transpose convolution on examples first
def test_conv2d_transpose(img_size, filter_size):
    print("Transpose convolution test for img_size={}, filter_size={}:".format(img_size, filter_size))
    
    x = (np.arange(img_size ** 2, dtype=np.float32) + 1).reshape((1, img_size, img_size, 1))
    f = (np.ones(filter_size ** 2, dtype=np.float32)).reshape((filter_size, filter_size, 1, 1))

    conv = tf.nn.conv2d_transpose \
        (x, f, output_shape = (1, img_size * 2, img_size * 2, 1), strides = [1, 2, 2, 1], padding = 'SAME')

    print("input:")
    print(x[0, :, :, 0])
    print("filter:")
    print(f[:, :, 0, 0])
    print("output:")
    print(conv[0, :, :, 0])

test_conv2d_transpose(img_size = 2, filter_size = 2)
test_conv2d_transpose(img_size = 2, filter_size = 3)
test_conv2d_transpose(img_size = 4, filter_size = 2)
test_conv2d_transpose(img_size = 4, filter_size = 3)
```

    Transpose convolution test for img_size=2, filter_size=2:
    input:
    [[1. 2.]
     [3. 4.]]
    filter:
    [[1. 1.]
     [1. 1.]]
    output:
    tf.Tensor(
    [[1. 1. 2. 2.]
     [1. 1. 2. 2.]
     [3. 3. 4. 4.]
     [3. 3. 4. 4.]], shape=(4, 4), dtype=float32)
    Transpose convolution test for img_size=2, filter_size=3:
    input:
    [[1. 2.]
     [3. 4.]]
    filter:
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    output:
    tf.Tensor(
    [[ 1.  1.  3.  2.]
     [ 1.  1.  3.  2.]
     [ 4.  4. 10.  6.]
     [ 3.  3.  7.  4.]], shape=(4, 4), dtype=float32)
    Transpose convolution test for img_size=4, filter_size=2:
    input:
    [[ 1.  2.  3.  4.]
     [ 5.  6.  7.  8.]
     [ 9. 10. 11. 12.]
     [13. 14. 15. 16.]]
    filter:
    [[1. 1.]
     [1. 1.]]
    output:
    tf.Tensor(
    [[ 1.  1.  2.  2.  3.  3.  4.  4.]
     [ 1.  1.  2.  2.  3.  3.  4.  4.]
     [ 5.  5.  6.  6.  7.  7.  8.  8.]
     [ 5.  5.  6.  6.  7.  7.  8.  8.]
     [ 9.  9. 10. 10. 11. 11. 12. 12.]
     [ 9.  9. 10. 10. 11. 11. 12. 12.]
     [13. 13. 14. 14. 15. 15. 16. 16.]
     [13. 13. 14. 14. 15. 15. 16. 16.]], shape=(8, 8), dtype=float32)
    Transpose convolution test for img_size=4, filter_size=3:
    input:
    [[ 1.  2.  3.  4.]
     [ 5.  6.  7.  8.]
     [ 9. 10. 11. 12.]
     [13. 14. 15. 16.]]
    filter:
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    output:
    tf.Tensor(
    [[ 1.  1.  3.  2.  5.  3.  7.  4.]
     [ 1.  1.  3.  2.  5.  3.  7.  4.]
     [ 6.  6. 14.  8. 18. 10. 22. 12.]
     [ 5.  5. 11.  6. 13.  7. 15.  8.]
     [14. 14. 30. 16. 34. 18. 38. 20.]
     [ 9.  9. 19. 10. 21. 11. 23. 12.]
     [22. 22. 46. 24. 50. 26. 54. 28.]
     [13. 13. 27. 14. 29. 15. 31. 16.]], shape=(8, 8), dtype=float32)



```python
def build_deep_autoencoder(img_shape, code_size):
    """PCA's deeper brother. See instructions above. Use `code_size` in layer definitions."""
    W, H, C = img_shape
    
    # encoder
    encoder = tf.keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', activation = 'elu'))
    encoder.add(L.MaxPooling2D(pool_size = (2, 2)))
    encoder.add(L.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'elu'))
    encoder.add(L.MaxPooling2D(pool_size = (2, 2)))
    encoder.add(L.Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'elu'))
    encoder.add(L.MaxPooling2D(pool_size = (2, 2)))
    encoder.add(L.Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'elu'))
    encoder.add(L.MaxPooling2D(pool_size = (2, 2)))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))

    # decoder
    decoder = tf.keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(2 * 2 * 256))
    decoder.add(L.Reshape((2, 2, 256)))
    decoder.add(L.Conv2DTranspose \
        (filters = 128, kernel_size = (3, 3), strides = 2, activation = 'elu', padding = 'same'))
    decoder.add(L.Conv2DTranspose \
        (filters = 64, kernel_size = (3, 3), strides = 2, activation = 'elu', padding = 'same'))
    decoder.add(L.Conv2DTranspose \
        (filters = 32, kernel_size = (3, 3), strides = 2, activation = 'elu', padding = 'same'))
    decoder.add(L.Conv2DTranspose \
        (filters = 3, kernel_size = (3, 3), strides = 2, activation = None, padding = 'same'))

    return encoder, decoder
```


```python
# Check autoencoder shapes along different code_sizes
get_dim = lambda layer: np.prod(layer.output_shape[1:])
for code_size in [1, 8, 32, 128, 512]:
    encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size = code_size)
    print("Testing code size %i" % code_size)
    assert encoder.output_shape[1:] == (code_size,),"encoder must output a code of required size"
    assert decoder.output_shape[1:] == IMG_SHAPE,   "decoder must output an image of valid shape"
    assert len(encoder.trainable_weights) >= 6,     "encoder must contain at least 3 layers"
    assert len(decoder.trainable_weights) >= 6,     "decoder must contain at least 3 layers"
    
    for layer in encoder.layers + decoder.layers:
        assert get_dim(layer) >= code_size, "Encoder layer %s is smaller than bottleneck (%i units)" % (layer.name, get_dim(layer))

print("All tests passed!")
```

    Testing code size 1
    Testing code size 8
    Testing code size 32
    Testing code size 128
    Testing code size 512
    All tests passed!



```python
# Look at encoder and decoder shapes.
# Total number of trainable parameters of encoder and decoder should be close.
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size = 32)
encoder.summary()
decoder.summary()
```

    Model: "sequential_12"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_20 (Conv2D)           (None, 32, 32, 32)        896       
    _________________________________________________________________
    max_pooling2d_20 (MaxPooling (None, 16, 16, 32)        0         
    _________________________________________________________________
    conv2d_21 (Conv2D)           (None, 16, 16, 64)        18496     
    _________________________________________________________________
    max_pooling2d_21 (MaxPooling (None, 8, 8, 64)          0         
    _________________________________________________________________
    conv2d_22 (Conv2D)           (None, 8, 8, 128)         73856     
    _________________________________________________________________
    max_pooling2d_22 (MaxPooling (None, 4, 4, 128)         0         
    _________________________________________________________________
    conv2d_23 (Conv2D)           (None, 4, 4, 256)         295168    
    _________________________________________________________________
    max_pooling2d_23 (MaxPooling (None, 2, 2, 256)         0         
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 1024)              0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 32)                32800     
    =================================================================
    Total params: 421,216
    Trainable params: 421,216
    Non-trainable params: 0
    _________________________________________________________________
    Model: "sequential_13"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_13 (Dense)             (None, 1024)              33792     
    _________________________________________________________________
    reshape_6 (Reshape)          (None, 2, 2, 256)         0         
    _________________________________________________________________
    conv2d_transpose_20 (Conv2DT (None, 4, 4, 128)         295040    
    _________________________________________________________________
    conv2d_transpose_21 (Conv2DT (None, 8, 8, 64)          73792     
    _________________________________________________________________
    conv2d_transpose_22 (Conv2DT (None, 16, 16, 32)        18464     
    _________________________________________________________________
    conv2d_transpose_23 (Conv2DT (None, 32, 32, 3)         867       
    =================================================================
    Total params: 421,955
    Trainable params: 421,955
    Non-trainable params: 0
    _________________________________________________________________


Convolutional autoencoder training. This will take **1 hour**. You're aiming at ~0.0056 validation MSE and ~0.0054 training MSE.


```python
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size = 32)

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = tf.keras.models.Model(inputs = inp, outputs = reconstruction)
autoencoder.compile(optimizer = "adamax", loss = 'mse')
```


```python
# we will save model checkpoints here to continue training in case of kernel death
model_filename = 'autoencoder.{0:03d}.hdf5'
last_finished_epoch = None

#### uncomment below to continue training from model checkpoint
#### fill `last_finished_epoch` with your latest finished epoch
# from keras.models import load_model
# s = reset_tf_session()
# last_finished_epoch = 4
# autoencoder = load_model(model_filename.format(last_finished_epoch))
# encoder = autoencoder.layers[1]
# decoder = autoencoder.layers[2]
```


```python
autoencoder.fit (
    x = X_train, y = X_train, epochs = 25,
    validation_data = [X_test, X_test],
    callbacks = [
        keras_utils.ModelSaveCallback(model_filename),
        #keras_utils.TqdmProgressCallback()
    ],
    verbose = 0,
    initial_epoch = last_finished_epoch or 0
)
```

    Model saved in autoencoder.000.hdf5
    Model saved in autoencoder.001.hdf5
    Model saved in autoencoder.002.hdf5
    Model saved in autoencoder.003.hdf5
    Model saved in autoencoder.004.hdf5
    Model saved in autoencoder.005.hdf5
    Model saved in autoencoder.006.hdf5
    Model saved in autoencoder.007.hdf5
    Model saved in autoencoder.008.hdf5
    Model saved in autoencoder.009.hdf5
    Model saved in autoencoder.010.hdf5
    Model saved in autoencoder.011.hdf5
    Model saved in autoencoder.012.hdf5
    Model saved in autoencoder.013.hdf5
    Model saved in autoencoder.014.hdf5
    Model saved in autoencoder.015.hdf5
    Model saved in autoencoder.016.hdf5
    Model saved in autoencoder.017.hdf5
    Model saved in autoencoder.018.hdf5
    Model saved in autoencoder.019.hdf5
    Model saved in autoencoder.020.hdf5
    Model saved in autoencoder.021.hdf5
    Model saved in autoencoder.022.hdf5
    Model saved in autoencoder.023.hdf5
    Model saved in autoencoder.024.hdf5





    <tensorflow.python.keras.callbacks.History at 0x7f623d985690>




```python
reconstruction_mse = autoencoder.evaluate(X_test, X_test, verbose = 0)
print("Convolutional autoencoder MSE:", reconstruction_mse)
for i in range(5):
    img = X_test[i]
    visualize(img, encoder, decoder)
```

    Convolutional autoencoder MSE: 0.005682000424712896



    
![png](images/autoencoder_output_24_1.png)
    



    
![png](images/autoencoder_output_24_2.png)
    



    
![png](images/autoencoder_output_24_3.png)
    



    
![png](images/autoencoder_output_24_4.png)
    



    
![png](images/autoencoder_output_24_5.png)
    



```python
# save trained weights
encoder.save_weights("encoder.h5")
decoder.save_weights("decoder.h5")
```


```python
# restore trained weights
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size = 32)
encoder.load_weights("encoder.h5")
decoder.load_weights("decoder.h5")

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = tf.keras.models.Model(inputs = inp, outputs = reconstruction)
autoencoder.compile(optimizer = "adamax", loss = 'mse')

print(autoencoder.evaluate(X_test, X_test, verbose = 0))
print(reconstruction_mse)
```

    0.005682000890374184
    0.005682000424712896


# Optional: Denoising Autoencoder

This part is **optional**, it shows you one useful application of autoencoders: denoising. You can run this code and make sure denoising works :) 

Let's now turn our model into a denoising autoencoder:
<img src="images/denoising.jpg" style="width:40%">

We'll keep the model architecture, but change the way it is trained. In particular, we'll corrupt its input data randomly with noise before each epoch.

There are many strategies to introduce noise: adding gaussian white noise, occluding with random black rectangles, etc. We will add gaussian white noise.


```python
def apply_gaussian_noise(X, sigma = 0.1):
    """
    adds noise from standard normal distribution with standard deviation sigma
    :param X: image tensor of shape [batch,height,width,3]
    Returns X + noise.
    """
    noise = np.random.normal(scale = sigma, size = X.shape)
    return X + noise
```


```python
# noise tests
theoretical_std = (X_train[:100].std() ** 2 + 0.5 ** 2) ** .5
our_std = apply_gaussian_noise(X_train[:100], sigma = 0.5).std()
assert abs(theoretical_std - our_std) < 0.01, "Standard deviation does not match it's required value. Make sure you use sigma as std."
assert abs(apply_gaussian_noise(X_train[:100], sigma = 0.5).mean() - X_train[:100].mean()) < 0.01, "Mean has changed. Please add zero-mean noise"
```


```python
# test different noise scales
plt.subplot(1, 4, 1)
show_image(X_train[0])
plt.subplot(1, 4, 2)
show_image(apply_gaussian_noise(X_train[:1], sigma = 0.01)[0])
plt.subplot(1, 4, 3)
show_image(apply_gaussian_noise(X_train[:1], sigma = 0.1)[0])
plt.subplot(1, 4, 4)
show_image(apply_gaussian_noise(X_train[:1], sigma = 0.5)[0])
```


    
![png](images/autoencoder_output_30_0.png)
    


Training will take **1 hour**.


```python
# we use bigger code size here for better quality
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size = 512)
assert encoder.output_shape[1:] == (512,), "encoder must output a code of required size"

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = tf.keras.models.Model(inp, reconstruction)
autoencoder.compile('adamax', 'mse')

for i in range(25):
    print("Epoch %i/25, Generating corrupted samples..." % (i + 1))
    X_train_noise = apply_gaussian_noise(X_train)
    X_test_noise = apply_gaussian_noise(X_test)
    
    # we continue to train our model with new noise-augmented data
    autoencoder.fit (
        x = X_train_noise, y = X_train, epochs = 1,
        validation_data = [X_test_noise, X_test],
        # callbacks=[keras_utils.TqdmProgressCallback()],
        verbose = 0
    )
```

    Epoch 1/25, Generating corrupted samples...
    Epoch 2/25, Generating corrupted samples...
    Epoch 3/25, Generating corrupted samples...
    Epoch 4/25, Generating corrupted samples...
    Epoch 5/25, Generating corrupted samples...
    Epoch 6/25, Generating corrupted samples...
    Epoch 7/25, Generating corrupted samples...
    Epoch 8/25, Generating corrupted samples...
    Epoch 9/25, Generating corrupted samples...
    Epoch 10/25, Generating corrupted samples...
    Epoch 11/25, Generating corrupted samples...
    Epoch 12/25, Generating corrupted samples...
    Epoch 13/25, Generating corrupted samples...
    Epoch 14/25, Generating corrupted samples...
    Epoch 15/25, Generating corrupted samples...
    Epoch 16/25, Generating corrupted samples...
    Epoch 17/25, Generating corrupted samples...
    Epoch 18/25, Generating corrupted samples...
    Epoch 19/25, Generating corrupted samples...
    Epoch 20/25, Generating corrupted samples...
    Epoch 21/25, Generating corrupted samples...
    Epoch 22/25, Generating corrupted samples...
    Epoch 23/25, Generating corrupted samples...
    Epoch 24/25, Generating corrupted samples...
    Epoch 25/25, Generating corrupted samples...



```python
X_test_noise = apply_gaussian_noise(X_test)
denoising_mse = autoencoder.evaluate(X_test_noise, X_test, verbose = 0)
print("Denoising MSE:", denoising_mse)
for i in range(5):
    img = X_test_noise[i]
    visualize(img, encoder, decoder)
```

    Denoising MSE: 0.0029018360655754805



    
![png](images/autoencoder_output_33_1.png)
    



    
![png](images/autoencoder_output_33_2.png)
    



    
![png](images/autoencoder_output_33_3.png)
    



    
![png](images/autoencoder_output_33_4.png)
    



    
![png](images/autoencoder_output_33_5.png)
    


# Optional: Image retrieval with autoencoders

So we've just trained a network that converts image into itself imperfectly. This task is not that useful in and of itself, but it has a number of awesome side-effects. Let's see them in action.

First thing we can do is image retrieval aka image search. We will give it an image and find similar images in latent space:

<img src="images/similar_images.jpg" style="width:60%">

To speed up retrieval process, one should use Locality Sensitive Hashing on top of encoded vectors. This [technique](https://erikbern.com/2015/07/04/benchmark-of-approximate-nearest-neighbor-libraries.html) can narrow down the potential nearest neighbours of our image in latent space (encoder code). We will caclulate nearest neighbours in brute force way for simplicity.


```python
# restore trained encoder weights
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size = 32)
encoder.load_weights("encoder.h5")
```


```python
images = X_train
codes = encoder(images)
assert len(codes) == len(images)
```


```python
from sklearn.neighbors.unsupervised import NearestNeighbors
nei_clf = NearestNeighbors(metric = "euclidean")
nei_clf.fit(codes)
```

    /home/bacti/anaconda3/envs/tensor/lib/python3.7/site-packages/sklearn/utils/deprecation.py:143: FutureWarning: The sklearn.neighbors.unsupervised module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
      warnings.warn(message, FutureWarning)





    NearestNeighbors(metric='euclidean')




```python
def get_similar(image, n_neighbors = 5):
    assert image.ndim == 3, "image must be [batch,height,width,3]"

    code = encoder.predict(image[None])
    (distances,), (idx,) = nei_clf.kneighbors(code, n_neighbors = n_neighbors)
    
    return distances, images[idx]
```


```python
def show_similar(image):
    distances, neighbors = get_similar(image, n_neighbors = 3)
    
    plt.figure(figsize = [8, 7])
    plt.subplot(1, 4, 1)
    show_image(image)
    plt.title("Original image")
    
    for i in range(3):
        plt.subplot(1, 4, i + 2)
        show_image(neighbors[i])
        plt.title("Dist=%.3f" % distances[i])
    plt.show()
```

Cherry-picked examples:


```python
# smiles
show_similar(X_test[247])
```


    
![png](images/autoencoder_output_41_0.png)
    



```python
# ethnicity
show_similar(X_test[56])
```


    
![png](images/autoencoder_output_42_0.png)
    



```python
# glasses
show_similar(X_test[63])
```


    
![png](images/autoencoder_output_43_0.png)
    


# Optional: Cheap image morphing


We can take linear combinations of image codes to produce new images with decoder.


```python
# restore trained encoder weights
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size = 32)
encoder.load_weights("encoder.h5")
decoder.load_weights("decoder.h5")
```


```python
for _ in range(5):
    image1, image2 = X_test[np.random.randint(0, len(X_test), size = 2)]
    code1, code2 = encoder.predict(np.stack([image1, image2]))

    plt.figure(figsize = [10, 4])
    for i,a in enumerate(np.linspace(0, 1, num = 7)):
        output_code = code1 * (1 - a) + code2 * (a)
        output_image = decoder.predict(output_code[None])[0]

        plt.subplot(1, 7, i + 1)
        show_image(output_image)
        plt.title("a=%.2f" % a)
        
    plt.show()
```


    
![png](images/autoencoder_output_47_0.png)
    



    
![png](images/autoencoder_output_47_1.png)
    



    
![png](images/autoencoder_output_47_2.png)
    



    
![png](images/autoencoder_output_47_3.png)
    



    
![png](images/autoencoder_output_47_4.png)
    


That's it!

Of course there's a lot more you can do with autoencoders.

If you want to generate images from scratch, however, we recommend you our honor track on Generative Adversarial Networks or GANs.
