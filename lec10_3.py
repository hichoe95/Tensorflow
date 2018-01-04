import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

training_epochs = 15

batch_size = 100

keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.get_variable("W1",shape = [784,256])
b1 = tf.Variable(tf.random_normal([256]))
l1 = tf.nn.relu(tf.matmul(x,w1) + b1)
l1 = tf.nn.dropout(l1, keep_prob=keep_prob)

w2 = tf.get_variable("W2", shape = [256,256])
b2 = tf.Variable(tf.random_normal([256]))
l2 = tf.nn.relu(tf.matmul(l1,w2) + b2)
l2 = tf.nn.dropout(l2, keep_prob=keep_prob)


w3 = tf.get_variable("W3", shape = [256,256])
b3 = tf.Variable(tf.random_normal([256]))
l3 = tf.nn.relu(tf.matmul(l2,w3) + b3)
l3 = tf.nn.dropout(l3, keep_prob=keep_prob)

w4 = tf.get_variable("W4", shape = [256,256])
b4 = tf.Variable(tf.random_normal([256]))
l4 = tf.nn.relu(tf.matmul(l3,w4) + b4)
l4 = tf.nn.dropout(l4, keep_prob=keep_prob)


w5 = tf.get_variable("W5", shape = [256,10])
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(l4,w5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= hypothesis, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_xs, y: batch_ys, keep_prob : 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print("Epoch:", "%04d" % (epoch +1), 'cost =', '{:.9f}'.format(avg_cost))

print("Learning Finished!")

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob : 1}))

