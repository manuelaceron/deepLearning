import tensorflow as tf

#1.	Building the computational graph
x= tf.constant(2.0)
w= tf.constant(-3.0)
b= tf.constant(-1.5)

SalidaAux=tf.multiply(x,w)
Neta=tf.add(SalidaAux,b)
Salida=tf.nn.relu(Neta)

#Salida=Neta
#Salida=tf.sigmoid(Neta)
#Salida=tf.tanh(Neta)
#Salida=x*w+b

#2.	Running the computational graph.
sess = tf.Session()
Valor_Salida=sess.run(Salida)

print("Salida: %s"%(Valor_Salida))
