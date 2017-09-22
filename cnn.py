import tensorflow as tf
import numpy as np
import cv2
import glob
import os
import time

def loadImages(trainPath, imageSize, classes = ["dogs", "cats"]):
    """Load images the path to the folder that contains
    the images and do resize and normalization. Return images, its labels and the name of images.

    trainPath: path to folder that contains images to train
    imageSize: the size used in cv2.resize()
    classes: types of images to train for
    """
    images = []
    labels = []
    imageNames = []

    for className in classes:
        index = classes.index(className)
        path = os.path.join(trainPath, className, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (imageSize, imageSize),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            imageNames.append(fl)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, imageNames

def create_weights(shape):
    """Return weights used to train the convolution layer in cnn.
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    """Return biases used to train the convolution layer in cnn.
    """
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input, numOfChannels, sizeOfFilter, numOfFilters):
    """Returns a constructed convolutional layer.

    numOfChannels: number of channels. For RGB it's 3.
    sizeOfFilter: size of filter used in convolutional layer.
    numOfFilters: number of filters (feature maps used in the convolutional lyaer)
    """
    # randomly initialize weights and bias to train for this layer
    weights = create_weights([sizeOfFilter, sizeOfFilter, numOfChannels, numOfFilters])
    biases = create_biases(numOfFilters)
    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases
    # max-pooling.
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    # activating with relu
    layer = tf.nn.relu(layer)
    return layer

def create_flatten_layer(layer):
    """Flattern convolutional layer and return it.
    """
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    """Create a fully connected layer.
    """
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

# set random seed
np.random.seed(1)
tf.set_random_seed(1)
# load images for training
trainPath = "train"
imageSize = 64
images, labels, imageNames = loadImages(trainPath, imageSize)

classes = ["dogs", "cats"]
numOfChannels = 3
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, imageSize,imageSize,numOfChannels], name='x')
trueY = tf.placeholder(tf.float32, shape=[None, len(classes)], name='trueY')

filterSize = 3
# num of filters in convolution layer 1,2,3
numOfFiltersConv1 = 16
numOfFiltersConv2 = 16
numOfFiltersConv3 = 32
# size of fully connected layer
sizeOfFCLayer = 64
# construct 3 convolution layers
layer_conv1 = create_convolutional_layer(x,
               numOfChannels,
               filterSize,
               numOfFiltersConv1)
layer_conv2 = create_convolutional_layer(layer_conv1,
               numOfFiltersConv1,
               filterSize,
               numOfFiltersConv2)
layer_conv3= create_convolutional_layer(layer_conv2,
               numOfFiltersConv2,
               filterSize,
               numOfFiltersConv3)
# flattern output from convolution layer
layer_flat = create_flatten_layer(layer_conv3)
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=sizeOfFCLayer,
                     use_relu=True)
layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=sizeOfFCLayer,
                     num_outputs=len(classes),
                     use_relu=False)
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=trueY)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
session.run(tf.global_variables_initializer())
# start training
saver = tf.train.Saver()
def train(num_iteration):
    # hardcoding 4 because we are only training for four images
    runtimes = np.zeros((num_iteration, 4))
    for i in range(num_iteration):
        for j in range(4):
            feed_dict_tr = {x: images[j].reshape((1, 64, 64, 3)),
                               trueY: labels[j].reshape((1, 2))}
            st = time.time()
            session.run(optimizer, feed_dict=feed_dict_tr)
            runtimes[i, j] = time.time() - st
        saver.save(session, 'dogs-cats-model')
    with open('runtime', 'w') as f:
        for i in range(runtimes.shape[0]):
            f.write("{}\n".format(runtimes[i]))
train(num_iteration=10)

testPath = "test"
imageSize = 64
images, labels, imageNames = loadImages(testPath, imageSize)

sess = tf.Session()
saver = tf.train.import_meta_graph('dogs-cats-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")
for i in range(len(imageNames)):
    feed_dict_testing = {x: images[i].reshape((1, 64, 64, 3)),
                                   trueY: labels[i].reshape((1, 2))}
    st = time.time()
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    print("{} takes {} to test".format(imageNames[i], time.time() - st))
    print("Probability that it's a dog: {}. Prob that it's a cat: {}\n".format(
        result[0][0], result[0][1]))
    print("TRUTH: it's a {}".format("dog" if "dog" in imageNames[i] else "cat"))
