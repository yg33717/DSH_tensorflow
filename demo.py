import os
import numpy as np
import tensorflow as tf
import cPickle

BATCH_SIZE = 200
LR = 0.001  # Learning rate
EPOCH = 600
LOAD_MODEL = False  # Whether or not continue train from saved model
TRAIN = True
HASHING_BITS = 12
CURRENT_DIR = os.getcwd()


def bias(name, shape, bias_start=0.0, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(
                              bias_start, dtype=dtype))
    return var


def weight(name, shape, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.random_normal_initializer(
                              stddev=stddev, dtype=dtype))
    return var


def fully_connected(value, output_shape, name='fully_connected', with_w=False):
    value = tf.reshape(value, [BATCH_SIZE, -1])
    shape = value.get_shape().as_list()

    with tf.variable_scope(name):
        weights = weight('weights', [shape[1], output_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)

    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases



def relu(value, name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)



def conv2d(value, output_dim, k_h=5, k_w=5,
           strides=[1, 1, 1, 1], name='conv2d'):
    with tf.variable_scope(name):
        weights = weight('weights',
                         [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv



def pool(value, k_size=[1,3,3,1],
           strides=[1, 2, 2, 1], name='pool1'):
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(value, ksize=k_size, strides=strides, padding='VALID')
        return pool

def pool_avg(value, k_size=[1,3,3,1],
           strides=[1, 2, 2, 1], name='pool1'):
    with tf.variable_scope(name):
        pool = tf.nn.avg_pool(value, ksize=k_size, strides=strides, padding='VALID')
        return pool

def lrn(value, depth_radius=1,alpha=5e-05,beta=0.75, name='lrn1'):
    with tf.variable_scope(name):
        norm1 = tf.nn.lrn(value, depth_radius=depth_radius, bias=1.0, alpha=alpha, beta=beta)
        return norm1




def discriminator(image, hashing_bits,reuse=False, name='discriminator'):
    with tf.name_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        conv1 = conv2d(image, output_dim=32, name='d_conv1')

        relu1 = relu(pool(conv1, name='d_pool1'), name='d_relu1')
        conv2 = conv2d(lrn(relu1,name='d_lrn1'), output_dim=32, name='d_conv2')
        relu2 = relu(pool_avg(conv2, name='d_pool2'), name='d_relu2')
        conv3 = conv2d(lrn(relu2, name='d_lrn2'), output_dim=64, name='d_conv3')
        pool3 = pool_avg(relu(conv3, name='d_relu3'), name='d_pool3')
        relu_ip1 = relu(fully_connected(pool3,output_shape=500, name='d_ip1'), name='d_relu4')
        ip2 = fully_connected(relu_ip1,output_shape=hashing_bits, name='d_ip2')

        return ip2



def read_cifar10_data():
    data_dir = CURRENT_DIR+'/data/cifar-10-batches-py/'
    train_name = 'data_batch_'
    test_name = 'test_batch'
    train_X = None
    train_Y = None
    test_X = None
    test_Y = None

    # train data
    for i in range(1,6):
        file_path = data_dir+train_name+str(i)
        with open(file_path, 'rb') as fo:
            dict = cPickle.load(fo)
            if  train_X is None:
                train_X = dict['data']
                train_Y = dict['labels']
            else:
                train_X = np.concatenate((train_X, dict['data']), axis=0)
                train_Y = np.concatenate((train_Y, dict['labels']), axis=0)
    # test_data
    file_path = data_dir + test_name
    with open(file_path, 'rb') as fo:
        dict = cPickle.load(fo)
        test_X = dict['data']
        test_Y = dict['labels']
    train_X = train_X.reshape((50000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float)
    # train_Y = train_Y.reshape((50000)).astype(np.float)
    test_X = test_X.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float)
    # test_Y.reshape((10000)).astype(np.float)

    train_y_vec = np.zeros((len(train_Y), 10), dtype=np.float)
    test_y_vec = np.zeros((len(test_Y), 10), dtype=np.float)
    for i, label in enumerate(train_Y):
        train_y_vec[i, int(train_Y[i])] = 1.  # y_vec[1,3] means #2 row, #4column
    for i, label in enumerate(test_Y):
        test_y_vec[i, int(test_Y[i])] = 1.  # y_vec[1,3] means #2 row, #4column

    return train_X/255., train_y_vec, test_X/255., test_y_vec

def hashing_loss(image,label,alpha,m):
    D = discriminator(image,HASHING_BITS)
    w_label = tf.matmul(label,label,False,True)
    
    r = tf.reshape(tf.reduce_sum(D*D,1),[-1,1])
    p2_distance = r - 2*tf.matmul(D,D,False,True)+tf.transpose(r)
    temp = w_label*p2_distance + (1-w_label)*tf.maximum(m-p2_distance,0)
    
    regularizer = tf.reduce_sum(tf.abs(tf.abs(D) - 1))
    d_loss = tf.reduce_sum(temp)/(BATCH_SIZE*(BATCH_SIZE-1)) + alpha * regularizer/BATCH_SIZE
    return d_loss


def train():
    train_dir  = CURRENT_DIR + '/logs/'
    global_step = tf.Variable(0, name='global_step', trainable=False)
    image = tf.placeholder(tf.float32, [BATCH_SIZE, 32,32,3], name='image')
    label = tf.placeholder(tf.float32, [BATCH_SIZE,10], name='label')

    alpha = tf.constant(0.01,dtype=tf.float32,name='tradeoff')
    # set m = 2*HASHING_BITS
    m = tf.constant(HASHING_BITS*2,dtype=tf.float32,name='bi_margin')
    d_loss_real = hashing_loss(image,label,alpha,m)
    
    
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]

    saver = tf.train.Saver()

    d_optim = tf.train.AdamOptimizer(LR, beta1=0.5) \
        .minimize(d_loss_real, var_list=d_vars, global_step=global_step)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.InteractiveSession(config=config)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    start = 0
    if LOAD_MODEL:
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(train_dir, ckpt_name))
            global_step = ckpt.model_checkpoint_path.split('/')[-1] \
                .split('-')[-1]
            print('Loading success, global_step is %s' % global_step)

        start = int(global_step)

    train_x, train_y, test_x, test_y = read_cifar10_data()

    for epoch in range(start,EPOCH):

        batch_idxs = 50000/BATCH_SIZE

        for idx in range(start, batch_idxs):
            image_idx = train_x[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            label_idx = train_y[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]

            sess.run([d_optim],feed_dict={image: image_idx, label:label_idx})
            # writer.add_summary(summary_str, idx + 1)

            errD_real = d_loss_real.eval(feed_dict={image: image_idx, label:label_idx})

            if (idx+1) % 10 == 0:
                print("[%3d/%3d][%4d/%4d] d_loss: %.8f" % (epoch+1,EPOCH,idx+1, batch_idxs,  errD_real))

        if (epoch+1) % 10 == 0:
            checkpoint_path = os.path.join(train_dir,'my_DSH_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch + 1)
            print '*********    model saved    *********'


    sess.close()

def toBinaryString(binary_like_values):
        numOfImage,bit_length = binary_like_values.shape
        list_string_binary = []
        for i in range(numOfImage):
            str = ''
            for j in range(bit_length):
                str += '0' if binary_like_values[i][j] <= 0 else '1'
            list_string_binary.append(str)
        return  list_string_binary

def evaluate():
    checkpoint_dir = CURRENT_DIR + '/logs/'     
    
    image = tf.placeholder(tf.float32, [BATCH_SIZE, 32, 32, 3], name='image')
    D = discriminator(image,HASHING_BITS)
    res = tf.sign(D)                     
    
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)                
    saver = tf.train.Saver(tf.all_variables())                          
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.InteractiveSession(config=config)

    train_x, train_y, test_x, test_y = read_cifar10_data()
    file_res = open('result.txt','w')
    # sys.stdout = file_res
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print('Loading success, global_step is %s' % global_step)
        for i in range(10000/BATCH_SIZE):
            eval_sess = sess.run(D, feed_dict={image: test_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE]})
           # print(eval_sess)
            w_res = toBinaryString(eval_sess)
            w_label = np.argmax(test_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE],axis=1)
            for j in range(BATCH_SIZE):
                file_res.write(w_res[j]+'\t'+str(w_label[j])+'\n')
        for i in range(50000/BATCH_SIZE):
            eval_sess = sess.run(D, feed_dict={image: train_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE]})
           # print(eval_sess)
            w_res = toBinaryString(eval_sess)
            w_label = np.argmax(train_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE],axis=1)
            for j in range(BATCH_SIZE):
                file_res.write(w_res[j]+'\t'+str(w_label[j])+'\n')


       # eval_sess = sess.run(res, feed_dict={image: test_x[:BATCH_SIZE]})
       # eval_sess = sess.run(res, feed_dict={image: test_x[:BATCH_SIZE]})
       # print(eval_sess)
    file_res.close()
    sess.close()

if __name__ == '__main__':

    if TRAIN:
        train()
    else:
        evaluate()
