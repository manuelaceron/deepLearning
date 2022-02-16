"""
Miniproyecto2
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import itertools
from sklearn.metrics import confusion_matrix

# Ruta de los archivos con los datos de entrenamiento y validación
DATA_FILE_TRAINING = 'data/forestTrain.csv'
DATA_FILE_TEST = 'data/forestTest.csv' 
DATA_FILE_PESOS = 'data/MLP_Pesos.csv'
      
#Definición de los arreglos que almacenarán los datos de entrenamiento
data=np.zeros([362,13],dtype=np.float64)
x_train=np.zeros([362,12],dtype=np.float64)
y_train=np.zeros([362,9],dtype=np.float64)
y_train_CM=np.zeros([362,1],dtype=np.float64)
Out_CM=np.zeros([362,1],dtype=np.float64)

#Lectura de los datos de entrenamiento desde el archivo
cont=0;
with open(DATA_FILE_TRAINING) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        data[cont,:]=(np.asarray(row))  
        x_train[cont,0] = data[cont,0]
        x_train[cont,1] = data[cont,1]        
        x_train[cont,2] = data[cont,2]
        x_train[cont,3] = data[cont,3]
        x_train[cont,4] = data[cont,4]
        x_train[cont,5] = data[cont,5]        
        x_train[cont,6] = data[cont,6]
        x_train[cont,7] = data[cont,7]
        x_train[cont,8] = data[cont,8]
        x_train[cont,9] = data[cont,9]        
        x_train[cont,10] = data[cont,10]
        x_train[cont,11] = data[cont,11]
        y_train_CM[cont]=data[cont,12]
#En archivo la clase viene como 0, 1 y 2. En este caso lo habitual es que quede una 
#neurona activa por clase. Es decir dependiendo de la clase solo se activará una #neurona
        if data[cont,12]==0:
               y_train[cont,0] =1
               y_train[cont,1] =0
               y_train[cont,2] =0
               y_train[cont,3] =0
               y_train[cont,4] =0
               y_train[cont,5] =0  
               y_train[cont,6] =0
               y_train[cont,7] =0
               y_train[cont,8] =0                   
        if data[cont,12]==1:
               y_train[cont,0] =0
               y_train[cont,1] =1
               y_train[cont,2] =0
               y_train[cont,3] =0
               y_train[cont,4] =0
               y_train[cont,5] =0  
               y_train[cont,6] =0
               y_train[cont,7] =0
               y_train[cont,8] =0      
        if data[cont,12]==2:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =1
               y_train[cont,3] =0
               y_train[cont,4] =0
               y_train[cont,5] =0  
               y_train[cont,6] =0
               y_train[cont,7] =0
               y_train[cont,8] =0    
        if data[cont,12]==3:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =0
               y_train[cont,3] =1
               y_train[cont,4] =0
               y_train[cont,5] =0  
               y_train[cont,6] =0
               y_train[cont,7] =0
               y_train[cont,8] =0    
        if data[cont,12]==4:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =0
               y_train[cont,3] =0
               y_train[cont,4] =1
               y_train[cont,5] =0  
               y_train[cont,6] =0
               y_train[cont,7] =0
               y_train[cont,8] =0    
        if data[cont,12]==5:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =0
               y_train[cont,3] =0
               y_train[cont,4] =0
               y_train[cont,5] =1  
               y_train[cont,6] =0
               y_train[cont,7] =0
               y_train[cont,8] =0    
        if data[cont,12]==6:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =0
               y_train[cont,3] =0
               y_train[cont,4] =0
               y_train[cont,5] =0  
               y_train[cont,6] =1
               y_train[cont,7] =0
               y_train[cont,8] =0    
        if data[cont,12]==7:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =0
               y_train[cont,3] =0
               y_train[cont,4] =0
               y_train[cont,5] =0  
               y_train[cont,6] =0
               y_train[cont,7] =1
               y_train[cont,8] =0    
        if data[cont,12]==8:
               y_train[cont,0] =0
               y_train[cont,1] =0
               y_train[cont,2] =0
               y_train[cont,3] =0
               y_train[cont,4] =0
               y_train[cont,5] =0  
               y_train[cont,6] =0
               y_train[cont,7] =0
               y_train[cont,8] =1                         
        cont=cont+1

n_samples=cont 
print(n_samples)   

# Definición de los place holders para los datos de entrenamiento y validación
# como se tiene una cantidad de datos diferentes para entrenar (120)  y para validar #(30)  se deja indefinido el tamaño 
X = tf.placeholder(tf.float32,(None, None), name='X')
Y = tf.placeholder(tf.float32,(None, None), name='Y')

# Definición de las variables para los pesos de la RNA
Wco = tf.Variable(tf.random_uniform((12,30),-1,1))
bco = tf.Variable(tf.random_uniform((30,),-1,1))

Wcs = tf.Variable(tf.random_uniform((30,9),-1,1))
bcs = tf.Variable(tf.random_uniform((9,),-1,1))

#Calculo de la salida.
#Capa oculta tangente sigmoidal
#Capa de salida sigmoidal
OutputCo= tf.tanh(tf.matmul(X,Wco)+bco)
Output=tf.nn.sigmoid((tf.matmul(OutputCo,Wcs)+bcs))

# Función de pérdida MSE
loss = tf.reduce_mean(tf.square(Y-Output))

# optimizador
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
# Entrenamiento de la red
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init) 

# Se entrena la red 5000 iteracioens
for i in range(5000):
  sess.run(train, {X: x_train, Y: y_train})

# Se prueba la red con los datos de entrenamiento
curr_loss, curr_Output = sess.run([loss,Output], {X:  x_train, Y: y_train})
print("Loss: %s Output: %s "%(curr_loss,curr_Output))

#Para la matriz de confusión se necesita la posición de la neurona que tuvo mayor #activación, esto determina la clase
for i in range(0,362):
    Out_CM[i] = np.argmax(curr_Output[i,:])

#Nombres de las clases para la matriz de confusión    
class_names=['Rango 1', 'Rango 2', 'Rango 3', 'Rango 4', 'Rango 5', 'Rango 6', 'Rango 7', 'Rango 8', 'Rango 9']

#Función que permite graficar la matriz de confusión
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(" Matrix de confusion Normalizada ")
    else:
        print('Matrix de confusion No Normalizada')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Salida Deseada')
    plt.xlabel('Salida Estimada')

# Calculo de la matriz de confusión
cnf_matrix = confusion_matrix(y_train_CM, Out_CM)
np.set_printoptions(precision=2)

# Graficación de la matriz de confusión no normalizada 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Matrix de confusion No Normalizada')

# Graficación de la matriz de confusión normalizada
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Matrix de confusion Normalizada')

plt.show()

#Definición de los arreglos que almacenaráan los datos de validación
data2=np.zeros([155,13],dtype=np.float64)
x_test=np.zeros([155,12],dtype=np.float64)
y_test=np.zeros([155,9],dtype=np.float64)
y_test_CM=np.zeros([155,1],dtype=np.float64)
Out_test_CM=np.zeros([155,1],dtype=np.float64)
#Lectura de los datos de entrenamiento desde el archivo
cont=0;
with open(DATA_FILE_TEST) as csvfile:
    
    readCSV = csv.reader(csvfile, delimiter=',')
    
    for row in readCSV:
        data2[cont,:]=(np.asarray(row))  
        x_test[cont,0] = data2[cont,0]
        x_test[cont,1] = data2[cont,1]        
        x_test[cont,2] = data2[cont,2]
        x_test[cont,3] = data2[cont,3]
        x_test[cont,4] = data2[cont,4]
        x_test[cont,5] = data2[cont,5]        
        x_test[cont,6] = data2[cont,6]
        x_test[cont,7] = data2[cont,7]
        x_test[cont,8] = data2[cont,8]
        x_test[cont,9] = data2[cont,9]        
        x_test[cont,10] = data2[cont,10]
        x_test[cont,11] = data2[cont,11]
        y_test_CM[cont]=data2[cont,12]
        if data2[cont,12]==0:
               y_test[cont,0] =1
               y_test[cont,1] =0
               y_test[cont,2] =0
               y_test[cont,3] =0
               y_test[cont,4] =0
               y_test[cont,5] =0  
               y_test[cont,6] =0
               y_test[cont,7] =0
               y_test[cont,8] =0                       
        if data2[cont,12]==1:
               y_test[cont,0] =0
               y_test[cont,1] =1
               y_test[cont,2] =0
               y_test[cont,3] =0
               y_test[cont,4] =0
               y_test[cont,5] =0  
               y_test[cont,6] =0
               y_test[cont,7] =0
               y_test[cont,8] =0  
        if data2[cont,12]==2:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =1
               y_test[cont,3] =0
               y_test[cont,4] =0
               y_test[cont,5] =0  
               y_test[cont,6] =0
               y_test[cont,7] =0
               y_test[cont,8] =0
        if data2[cont,12]==3:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =0
               y_test[cont,3] =1
               y_test[cont,4] =0
               y_test[cont,5] =0  
               y_test[cont,6] =0
               y_test[cont,7] =0
               y_test[cont,8] =0    
        if data2[cont,12]==4:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =0
               y_test[cont,3] =0
               y_test[cont,4] =1
               y_test[cont,5] =0  
               y_test[cont,6] =0
               y_test[cont,7] =0
               y_test[cont,8] =0    
        if data2[cont,12]==5:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =0
               y_test[cont,3] =0
               y_test[cont,4] =0
               y_test[cont,5] =1  
               y_test[cont,6] =0
               y_test[cont,7] =0
               y_test[cont,8] =0    
        if data2[cont,12]==6:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =0
               y_test[cont,3] =0
               y_test[cont,4] =0
               y_test[cont,5] =0  
               y_test[cont,6] =1
               y_test[cont,7] =0
               y_test[cont,8] =0    
        if data2[cont,12]==7:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =0
               y_test[cont,3] =0
               y_test[cont,4] =0
               y_test[cont,5] =0  
               y_test[cont,6] =0
               y_test[cont,7] =1
               y_test[cont,8] =0
        if data2[cont,12]==8:
               y_test[cont,0] =0
               y_test[cont,1] =0
               y_test[cont,2] =0
               y_test[cont,3] =0
               y_test[cont,4] =0
               y_test[cont,5] =0  
               y_test[cont,6] =0
               y_test[cont,7] =0
               y_test[cont,8] =1                        
        cont=cont+1
    #    print(cont)
        
n_samples=cont 
print(n_samples)   

# Se prueba la red con los datos de validación
curr_loss, curr_Output = sess.run([loss,Output], {X:  x_test, Y: y_test})
print("Loss: %s Output: %s "%(curr_loss,curr_Output))

for ii in range(0,155):
    Out_test_CM[ii] = np.argmax(curr_Output[ii,:])
    
   
# Calculo de la matriz de confusión
cnf_matrix = confusion_matrix(y_test_CM, Out_test_CM)
np.set_printoptions(precision=2)

# Graficación de la matriz de confusión no normalizada 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Matrix de confusion No Normalizada')

# Graficación de la matriz de confusión normalizada 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Matrix de confusion Normalizada')

plt.show()


#with open(DATA_FILE_PESOS, 'w', newline='') as csvfile:
#    weightwriter = csv.writer(csvfile, delimiter=',',
#                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    weightwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
#    weightwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
