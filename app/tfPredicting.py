import tensorflow as tf
import prepare

inputSize = 12
L1Size = 8
outputSize = 4

# declare the training data placeholders
x = tf.placeholder(tf.float64, [None, inputSize])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float64, [None, outputSize])

# now declare the weights connecting the input to the hidden layer
W1 = tf.get_variable('W1', shape=[inputSize, L1Size], dtype=tf.float64)
b1 = tf.get_variable('b1', shape=[L1Size], dtype=tf.float64)
# and the weights connecting the hidden layer to the output layer
W2 = tf.get_variable('W2', shape=[L1Size, outputSize], dtype=tf.float64)
b2 = tf.get_variable('b2', shape=[outputSize], dtype=tf.float64)

# calculate the output of the hidden layer
L1 = tf.add(tf.matmul(x, W1), b1)
L1 = tf.nn.relu(L1)
# L1 = tf.nn.sigmoid(L1)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.add(tf.matmul(L1, W2), b2)
y_ = tf.nn.softmax(y_)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

saver = tf.train.Saver()

def predict(inputMatrix, outputMatrix):
    with tf.Session() as sess:
        saver.restore(sess, "./model.ckpt")     # TODO wpisywanie sciezki
        print(sess.run(accuracy, feed_dict={x: inputMatrix, y: outputMatrix}))