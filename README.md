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

Prepare image data for training and validation:

> unzip train.zip and validation.zip under data directory
