Implements a two-layer neural networks from scratch, with backpropagation. This includes the forward and backward
pass of a linear layer (with ReLU activations), a Softmax classifier with gradients and dropout as regularizer to the network. 

(Assignment from Course 395 Machine Learning - Imperial College London)

=====================================


### Required packages
* pandas
* keras v 2.1.4
* tensorflow 1.6.0
* Pillow
* bayesian-optimization
* h5py 
* scikit-image

### IMPORTANT
Currently 'datasets' is a symbolic link to a shared copy of the data in vol/bitbucket/ in the lab. If any of the code which requires the data is run on a non-lab computer it will crash. To solve this, please delete the symbolic link and replace it with a a directory named 'datasets' containing the data as originally planned.

### Running test.py for question 5
The function test_fer_model takes three parameters:
 * img_folder # where the test images are stored
 * model="fer_model.pickle" # location of the model pickle
 * mean_image_file='mean_image.pickle' # location of the mean training image pickle
Depending on where this function is called the defaults for the pickle files may not work and they will need to be entered manually. These files are located in assignment2_advanced/src/.

### Running test.py for CNN
[test.py](../assignment2_advanced/src/test.py) is setup as a unittest file. Please, if that is not intended, 
 take the file [cnn_test.py](../assignment2_advanced/src/cnn_test.py) instead.

For running the model with your test data, please either delete the [Test](../assignment2_advanced/data/Test)
and [labels](../assignment2_advanced/data/labels) folders and put new Test and labels folders with all your images 
exactly in the same place or specify the paths to your data in test.py accordingly. 

The functions in test.py and cnn_test.py also do the appropriate preprocessing for testing. 
