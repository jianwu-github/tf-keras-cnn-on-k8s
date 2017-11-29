### Training TensorFlow/Keras CNN Model with image datasets like dogs-vs-cats and Food-101 on Kubernetes Cluster

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
$ python3 tf_keras_cnn_image_classifier.py -ep 5 -bs 16
Using TensorFlow backend.
Found 2000 images belonging to 2 classes.
Found 800 images belonging to 2 classes.
Epoch 1/5
2017-11-22 11:21:09.881558: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
125/125 [==============================] - 51s 412ms/step - loss: 0.7084 - acc: 0.5305 - val_loss: 0.6821 - val_acc: 0.5150
Epoch 2/5
125/125 [==============================] - 54s 431ms/step - loss: 0.6646 - acc: 0.5910 - val_loss: 0.6385 - val_acc: 0.6338
Epoch 3/5
125/125 [==============================] - 65s 522ms/step - loss: 0.6404 - acc: 0.6520 - val_loss: 0.5794 - val_acc: 0.6813
Epoch 4/5
125/125 [==============================] - 61s 485ms/step - loss: 0.6044 - acc: 0.6685 - val_loss: 0.6015 - val_acc: 0.6613
Epoch 5/5
125/125 [==============================] - 56s 449ms/step - loss: 0.5727 - acc: 0.6995 - val_loss: 0.5359 - val_acc: 0.7412

```

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
