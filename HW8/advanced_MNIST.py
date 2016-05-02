import tensorflow as tf
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