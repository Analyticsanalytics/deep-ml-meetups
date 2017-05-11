import tensorflow as tf
from scipy.stats import logistic
from scipy.special import expit
print logistic.cdf(0.387)
print expit(0.387)

def softmax(z):
    # z is a vector
    return np.exp(z) / np.sum(np.exp(z))
 
def sigmoid(x):
    # x can be a vector
    return 1.0/(1.0+np.exp(-x))
 
def sigmoid_gradient(x):
    # x can be a vector
    return sigmoid(x)*(1-sigmoid(x))
 
def ReLU(x):
    # x can be a vector
    return np.maximum(x,0)
 
def ReLU_gradient(x):
    # x can be a vector
    return 1.0*(x>0)

def ddsigmoid(z):
  return expit(z) * (1.0 - expit(z))

print ddsigmoid(0.387)

sess = tf.InteractiveSession()
tf.reset_default_graph()

one=tf.constant(1.0)
X = tf.placeholder("float") # create symbolic variable
Y = tf.placeholder("float") # create symbolic variable

x_77=tf.constant(0.387)

# derivative of sigmoid= sigmoid(y) * (1.0 - sigmoid(y))

sigmoid=(tf.div(one, (one + tf.exp(-X))))
dsigmoid=tf.multiply(Y, tf.subtract(one,Y)) 
  
init = tf.initialize_all_variables() # you need to initialize variables (in this case just variable W)
sess = tf.Session()
print sess.run(init)

print sess.run(sigmoid, {X:x_77.eval(session=sess)}) 
print sess.run(sigmoid, {X:0.387}) 
print sess.run(dsigmoid,{Y:sigmoid.eval({X:0.387}, session=sess)}) 
