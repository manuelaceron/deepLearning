import tensorflow as tf

#1.	Building the computational graph
#x= tf.constant(2.0)
#w= tf.constant(-3.0)
#b= tf.constant(-1.5)

x = tf.placeholder(tf.float32)
w = tf.Variable([-3.0], dtype=tf.float32)
b = tf.Variable([-1.5], dtype=tf.float32)

Neta=tf.multiply(x,w)+b
Salida=tf.nn.relu(Neta)

#Salida=Neta
#Salida=tf.sigmoid(Neta)
#Salida=tf.tanh(Neta)
#Salida=x*w+b

#2.	Running the computational graph.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) 

Valor_Salida=sess.run(Salida,{x: [-5, -2, 0, 2, 5]})

print("Salida: %s"%(Valor_Salida))
