import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.python.client import device_lib



print "------- Available GPUs ---------"
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

#get_available_gpus()



print "------- Logging device ---------"
def print_logging_device():
    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print sess.run(c)

#print_logging_device()



print "------- Multiplication on gpu0 vs cpu ---------"
def matrix_mul(device_name, matrix_sizes):
    time_values = []
    #device_name = "/cpu:0"
    for size in matrix_sizes:
        with tf.device(device_name):
            random_matrix = tf.random_uniform(shape=(2,2), minval=0, maxval=1)
            dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
            sum_operation = tf.reduce_sum(dot_operation)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
            startTime = datetime.now()
            result = session.run(sum_operation)
        td = datetime.now() - startTime
        time_values.append(td.microseconds/1000)
        print ("matrix shape:" + str(size) + "  -- time: "+str(td.microseconds/1000))
    return time_values


#matrix_sizes = range(100,3000,100)
#time_values_gpu = matrix_mul("/gpu:0", matrix_sizes)
#time_values_cpu = matrix_mul("/cpu:0", matrix_sizes)
#print ("GPU time" +  str(time_values_gpu))
#print ("CPUtime" + str(time_values_cpu))
print "--------------------------------"


print "------- Multi GPU ---------"
def multi_gpu():
    c = []
    for d in ['/cpu','/gpu:0', '/gpu:1','/gpu:2', '/gpu:3']:
      with tf.device(d):
        a = tf.ones(shape=[3,3], dtype=tf.float32)
        b = tf.ones(shape=[3,3], dtype=tf.float32)
        c.append(tf.matmul(a, b))
    with tf.device('/cpu:0'):
      sum = tf.add_n(c)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(sum))
    
#multi_gpu()
print "--------------------------------"
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_MultiGPU/multigpu_basics.py

def single_gpu_sum(matrix_size ,n):
    
    # Create a graph to store results
    comp = []

    def matpow(M, n):
        if n < 1: 
            return M
        else:
            return tf.matmul(M, matpow(M, n-1))

    for d in ['/gpu:0', '/gpu:1','/gpu:2', '/gpu:3']:
        with tf.device('/gpu:0'):
            A = np.random.rand(matrix_size, matrix_size).astype('float32')
        
            # Compute A^n and B^n and store results in c1
            comp.append(matpow(A, n))

    with tf.device('/cpu:0'):
      sum = tf.add_n(comp) #Addition of all elements in c1, i.e. A^n + B^n

    startTime = datetime.now()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(sum)
    td = datetime.now() - startTime
    #print ("n:" + str(n) + "  -- time: "+str(td))
    return td

def multi_gpu_sum(matrix_size ,n):
    
    # Create a graph to store results
    comp = []

    def matpow(M, n):
        if n < 1: 
            return M
        else:
            return tf.matmul(M, matpow(M, n-1))
        
    for d in ['/gpu:0', '/gpu:1','/gpu:2', '/gpu:3']:
        with tf.device(d):
            A = np.random.rand(matrix_size, matrix_size).astype('float32')
            comp.append(matpow(A, n))

    with tf.device('/cpu:0'):
      sum = tf.add_n(comp) #Addition of all elements in c1, i.e. A^n + B^n

    startTime = datetime.now()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(sum)
    td = datetime.now() - startTime
    #print ("n:" + str(n) + "  -- time: "+str(td))
    return td
    



single_gpu = str(single_gpu_sum(1000 ,10))
multi_gpu = str(multi_gpu_sum(1000 ,10))
print single_gpu
print multi_gpu
