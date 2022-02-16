import tensorflow as tf

# training data
x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[0],[0],[1]]

# Model parameters
Wco = tf.Variable(tf.random_uniform((2,10),-1,1))
bco = tf.Variable(tf.random_uniform((10,),-1,1))

Wcs = tf.Variable(tf.random_uniform((10,1),-1,1))
bcs = tf.Variable(tf.random_uniform((1,),-1,1))


# Model input and output
x = tf.placeholder(tf.float32,(4,2))
y = tf.placeholder(tf.float32,(4,1))

OutputCo= tf.tanh(tf.matmul(x,Wco)+bco)
Output= tf.sigmoid(tf.matmul(OutputCo,Wcs)+bcs)


# loss
loss = tf.reduce_sum(tf.square(Output - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) 
for i in range(10000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss, curr_Output = sess.run([Wco, bco,Wcs, bcs, loss,Output], {x: x_train, y: y_train})
print("Wco: %s bco: %s Wcs: %s bcs: %s loss: %s Output: %s "%(curr_Wco, curr_bco, curr_Wcs, curr_bcs,curr_loss,curr_Output))