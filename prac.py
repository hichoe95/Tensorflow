import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#%matplotlib inline

batch_size = 1
img_height = 224
img_width= 224
img_channel = 3

x = tf.placeholder(tf.float32,[batch_size,img_height,img_width,img_channel]);

w = tf.Variable(tf.random_normal([5,5,3,64], stddev=0.35));
output = tf.nn.conv2d(x, w,strides=[1,2,2,1],padding='SAME');

sess = tf.Session();

with tf.Session() as sess:
	tf.global_variables_initializer().run();	img = np.array(Image.open('test.jpg'));
	plt.imshow(img)
	plt.show()
	img = img.reshape([1,img_height, img_width, img_channel])
	_out = sess.run(output,{x:img})
	print(_out.shape)

	plt.imshow(_out[0,:,:,0], cmap='gray')
	plt.show()
