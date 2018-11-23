import tensorflow as tf
import numpy as np

a = [0.6, 0.48, -0.37, -0.77, 0.7]
b = [1, 1, -1, -1, -1]


# compare_zero = np.zeros(())
a = np.array(a, np.float32)
b = np.array(b, np.float32)
bool_b = b > 0
bool_a = a > 0

correct = tf.equal(bool_a, bool_b)

# correct = tf.equal(tf.where(a > 0), tf.where(b > 0))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))
#
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# acc = sess.run(acc)
acc = sess.run(acc)
print(acc)