import tensorflow as tf
import input_data

#import MNIST
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#create interactive tensorflow session
sess = tf.InteractiveSession()

#Softmax regression
#x holds input image
#y holds output classes for each of the 10 digits (placeholders)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Weights and bias for the output layer
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#initialize variables in the session (W and b)
sess.run(tf.initialize_all_variables())

#softmax after matrix regression
y = tf.nn.softmax(tf.matmul(x,W) + b)

#Cross entropy cost
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#training with an alpha of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#load 50 training examples
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#evaluation
#Determine the number of correct predictors
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#cast booleans to 0's and 1's
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print accuracy
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))






###CONVOLUTIONAL NEURAL NETS

#Functions for weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#Strides of 1 and use 2*2 blocks
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#first layer (5*5 patch) with one input channel and 32 output channels
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#second and third are the image sizes and the 4th arg is number of color channels
x_image = tf.reshape(x, [-1,28,28,1])

#convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



#second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#Densely connected layer with all 1024 neurons
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#Implement Dropouts
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#Softmax Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)




#Evaluation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))