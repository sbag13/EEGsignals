import tensorflow as tf
import prepare, sys

# Python optimisation variables
learning_rate = 0.5
tf_epochs = 5

# Neural network parameters
inputSize = 12
L1Size = 8
outputSize = 4
std_dev = 0.03

# declare the training data placeholders
x = tf.placeholder(tf.float64, [None, inputSize])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float64, [None, outputSize])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([inputSize, L1Size], stddev=std_dev, dtype=tf.float64), name='W1')
b1 = tf.Variable(tf.random_normal([L1Size], dtype=tf.float64), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([L1Size, outputSize], stddev=std_dev, dtype=tf.float64), name='W2')
b2 = tf.Variable(tf.random_normal([outputSize], dtype=tf.float64), name='b2')

# calculate the output of the hidden layer
L1 = tf.add(tf.matmul(x, W1), b1)
L1 = tf.nn.relu(L1)
# L1 = tf.nn.sigmoid(L1)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.add(tf.matmul(L1, W2), b2)
y_ = tf.nn.softmax(y_)

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                        + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

saver = tf.train.Saver()
    
def startLearning():
    inputMatrix, outputMatrix = prepare.prepareAllSamples("./training_samples")

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_samples = int(len(inputMatrix))  
        for epoch in range(tf_epochs):
            avg_cost = 0
            for i in range(total_samples):
                sys.stdout.write("\r%d / %d   " % (i , total_samples))
                sys.stdout.flush()
                _, c = sess.run([optimiser, cross_entropy], 
                            feed_dict={x: inputMatrix, y: outputMatrix})
                avg_cost += c
            avg_cost /= total_samples
            print("\nEpoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        print(sess.run(accuracy, feed_dict={x: inputMatrix, y: outputMatrix}))
        save_path = saver.save(sess, "./model.ckpt")
        print("Model saved in path: %s" % save_path)
    