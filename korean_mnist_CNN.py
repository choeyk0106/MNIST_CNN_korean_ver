import tensorflow as tf
import random
import numpy as np
import h5py

tf.set_random_seed(777)  # reproducibility

def make_one_hot_vector(labels):
    num_labels = labels.shape[0]
    num_class = 14
    index_offset = np.arange(num_labels)*num_class
    label_one_hot = np.zeros([num_labels, num_class])
    label_one_hot.flat[index_offset+labels.ravel()] = 1
    return label_one_hot

def make_placeholder(tensor):
    tensor = np.reshape(tensor, [-1, 784])
    return tensor

def transpose_input_images(images):
    return np.reshape(images, [-1,28,28,1])

with h5py.File('resized_kalph_train.hf', 'r') as hf: 
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])
train_y_vector = make_one_hot_vector(labels)
train_x_image = transpose_input_images(images)

# read the test data
with h5py.File('resized_kalph_test.hf', 'r') as hf: 
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])
test_y_vector = make_one_hot_vector(labels)
test_x_image = transpose_input_images(images)

# hyper parameters
learning_rate = 0.001
training_epochs = 1
batch_size = 100
test_batch_size = 100

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 14])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
'''

# L3 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 128 * 4 * 4])
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 14],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([14]))
logits = tf.matmul(L4, W5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/tensorflowlogs", sess.graph)





train_batch_size = 19600
test_data_size = 3920

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(train_batch_size / batch_size)
    
    iterator = 0
    while iterator < train_batch_size:
        x_tensor = train_x_image[iterator:iterator+batch_size]
        x_batch = make_placeholder(x_tensor)
        y_batch = train_y_vector[iterator:iterator+batch_size]
        iterator += batch_size

        feed_dict = {X: x_batch, Y: y_batch, keep_prob:0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

iterator = 0
count = 0
max_accuracy, min_accuracy, mean_accuracy = 0, 1, 0
while  iterator < test_data_size:
    test_x_batch = make_placeholder(test_x_image[iterator:iterator+test_batch_size])
    test_y_batch = test_y_vector[iterator:iterator+test_batch_size]
    iterator += test_batch_size

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    prob = sess.run(accuracy, feed_dict={X: test_x_batch, Y: test_y_batch, keep_prob:1})
    count+=1
    """
    if prob >0.85:  
        print('test Accuracy:', prob)
        count+=1
        mean_accuracy += prob
    """
    if max_accuracy < prob:
        max_accuracy = prob
    if min_accuracy > prob:
        min_accuracy = prob

print("max accuracy : ", max_accuracy, "min accuracy : ", min_accuracy)
print("mean accuracy : ", mean_accuracy/(count))

test_x_batch = make_placeholder(test_x_image)
print()
print('---- 10 test case ----')
for iterator in range(10):
    r = random.randint(0, 3920 - 1)
    label = sess.run(tf.argmax(test_y_vector[r:r + 1], 1))
    predict = sess.run(tf.argmax(logits, 1), feed_dict={X: test_x_batch[r:r + 1], keep_prob:1})
    print("Label: ", label, "Prediction: ", predict)


