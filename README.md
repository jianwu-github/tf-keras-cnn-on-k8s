### Training TensorFlow/Keras CNN Model with image datasets like dogs-vs-cats and Food-101 on Kubernetes Cluster

#### Buidling CNN image classifier with Keras

[Learning Keras by building dogs-vs-cats image classifier](https://www.slideshare.net/jianwu/learning-keras-by-building-dogsvscats-image-classifier)

Setup TensorFlow/Keras and OpenCV for Python 3 dev environment on mac os:

```bash
$ sudo xcodebuild -license

$ sudo xcode-select --install

$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

$ brew update

$ brew tap homebrew/science

$ brew install python python3

$ brew linkapps python

$ brew linkapps python3

$ brew install opencv3

$ echo 'export PATH="/usr/local/opt/opencv/bin:$PATH"' >> ~/.bash_profile

$ echo /usr/local/opt/opencv/lib/python3.6/site-packages >> /usr/local/lib/python3.6/site-packages/openc3.pth

$ echo /usr/local/opt/opencv/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/openc3.pth

$ sudo pip3 install numpy scipy matplotlib scikit-learn pandas ipython

$ sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.4.0-py3-none-any.whl
```

Notes to use Keras 2 with TensorFlow 1.4 Backend:

> You should use Keras Functional API to build Neural Network Model(s)

Prepare image data for training and validation:

> unzip train.zip and validation.zip under data directory

To run tf_keras_cnn_image_classifier through command line:

```bash
$ $ python3 tf_keras_cnn_image_classifier.py -ep 10 -bs 20
Training Keras CNN Image Classifier: ==================================
  Training Data Dir:   data/train
  Validation Data Dir: data/validation
  Categories/Labels:   ['cats', 'dogs']
Found 2000 images belonging to 2 classes.
Found 800 images belonging to 2 classes.
2017-12-01 22:24:20.710246: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Epoch 1/10
100/100 [==============================] - 56s - loss: 0.7421 - acc: 0.5150 - val_loss: 0.6775 - val_acc: 0.5025
Epoch 2/10
100/100 [==============================] - 59s - loss: 0.6748 - acc: 0.6030 - val_loss: 0.6287 - val_acc: 0.6475
Epoch 3/10
100/100 [==============================] - 50s - loss: 0.6252 - acc: 0.6465 - val_loss: 0.5667 - val_acc: 0.6900
Epoch 4/10
100/100 [==============================] - 50s - loss: 0.5944 - acc: 0.6910 - val_loss: 0.5886 - val_acc: 0.6700
Epoch 5/10
100/100 [==============================] - 49s - loss: 0.5800 - acc: 0.7015 - val_loss: 0.5403 - val_acc: 0.7212
Epoch 6/10
100/100 [==============================] - 47s - loss: 0.5599 - acc: 0.7235 - val_loss: 0.9021 - val_acc: 0.5350
Epoch 7/10
100/100 [==============================] - 47s - loss: 0.5514 - acc: 0.7255 - val_loss: 0.5651 - val_acc: 0.6787
Epoch 8/10
100/100 [==============================] - 48s - loss: 0.5352 - acc: 0.7385 - val_loss: 0.5571 - val_acc: 0.7050
Epoch 9/10
100/100 [==============================] - 47s - loss: 0.5259 - acc: 0.7385 - val_loss: 0.5947 - val_acc: 0.6862
Epoch 10/10
100/100 [==============================] - 47s - loss: 0.5000 - acc: 0.7620 - val_loss: 0.4990 - val_acc: 0.7575
```

To test trained Keras CNN Model with test data under "data/test" directory, run tf_keras_cnn_image_predictor through command line:

```bash
$ python3 tf_keras_cnn_image_predictor.py
2017-12-01 22:40:28.951190: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA

Test Trained CNN Model with Dog Images: =======================
 Predicted Probability of Dog Image is [ 0.65954769]
 Predicted Probability of Dog Image is [ 0.99422461]
 Predicted Probability of Dog Image is [ 0.29893339]
 Predicted Probability of Dog Image is [ 0.73799419]
 Predicted Probability of Dog Image is [ 0.99334794]

Test Trained CNN Model with Cat Images: =======================
 Predicted Probability of Dog Image is [ 0.12118033]
 Predicted Probability of Dog Image is [ 0.0061747]
 Predicted Probability of Dog Image is [ 0.0073453]
 Predicted Probability of Dog Image is [ 0.45440799]
 Predicted Probability of Dog Image is [ 0.60804558]
 
```

#### Running TensorFlow CNN image classifier on Kubernetes cluster

To run tf_cnn_image_classifier through command line:

```bash
$ python3 tf_cnn_image_classifier.py -ep 5 -bs 30
[None, 17, 17, 32]
2017-11-27 12:17:14.719266: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
At 0th epoch: ======================================================
    At 32th step, the cost for training samples is 0.6041718125343323
    At 64th step, the cost for training samples is 0.691839873790741
    the cost for validation samples is 0.6904814839363098

At 1th epoch: ======================================================
    At 32th step, the cost for training samples is 0.6702802777290344
    At 64th step, the cost for training samples is 0.5561310648918152
    the cost for validation samples is 0.658843994140625

At 2th epoch: ======================================================
    At 32th step, the cost for training samples is 0.7262727618217468
    At 64th step, the cost for training samples is 0.3656562268733978
    the cost for validation samples is 0.7333311438560486

At 3th epoch: ======================================================
    At 32th step, the cost for training samples is 0.45142433047294617
    At 64th step, the cost for training samples is 0.15599210560321808
    the cost for validation samples is 0.670315682888031

At 4th epoch: ======================================================
    At 32th step, the cost for training samples is 0.4887373149394989
    At 64th step, the cost for training samples is 0.11935045570135117
    the cost for validation samples is 0.9625208973884583
```
