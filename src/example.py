import tensorflow as tf

x = tf.placeholder(tf.float32, name="x")
y_ = tf.placeholder(tf.float32)
d_ = tf.placeholder(tf.float32)

W = tf.Variable([0], dtype=tf.float32)
b = tf.Variable([0], dtype=tf.float32)
y = W * x + b

d = tf.gradients(y,x)[0]

loss = tf.square(y-y_) + tf.square(d-d_)

train_op = tf.train.AdamOptimizer(100).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess.run([d], feed_dict={'x:0':2})

for i in range(300):
    sess.run(train_op, feed_dict={x:1, y_:3, d_: 0.2})
    _w, _b = sess.run([W, b])
    print(_w, _b)
