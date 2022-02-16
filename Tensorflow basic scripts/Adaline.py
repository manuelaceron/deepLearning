import tensorflow as tf

# training data
x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[0],[0],[1]]

# Model parameters
W = tf.Variable(tf.random_uniform((2,1),-1,1))
b = tf.Variable(tf.random_uniform((1,),-1,1))

# Model input and output
x = tf.placeholder(tf.float32,(4,2))
y = tf.placeholder(tf.float32,(4,1))
Output= tf.matmul(x,W)+b

# loss
loss = tf.reduce_sum(tf.square(Output - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) 
for i in range(500):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss, curr_Output = sess.run([W, b, loss,Output], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s Output: %s "%(curr_W, curr_b, curr_loss,curr_Output))