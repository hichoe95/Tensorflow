import tensorflow as tf
import matplotlib.pyplot as plt

#x_data = [1,2,3]
#y_data = [1,2,3]

#W = tf.placeholder(tf.float32);
#W = tf.Variable(tf.random_normal([1]), name='weight');
#X = tf.placeholder(tf.float32);
#Y = tf.placeholder(tf.float32);
X = [1,2,3]
Y = [1,2,3]
W = tf.Variable(5.0);
hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis - Y));

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1);
train = optimizer.minimize(cost);

sess = tf.Session();

sess.run(tf.global_variables_initializer());

for step in range(100):
    print(step, sess.run([W]));
    sess.run(train);

