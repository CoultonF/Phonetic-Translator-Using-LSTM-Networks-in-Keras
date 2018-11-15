from __future__ import print_function
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os, time, itertools, imageio, pickle, random, string
import tensorflow as tf
from pandas import *
from numpy import array

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

# G(z)
# z shape:       (?,28,28,1)
# y_label shape: (?,29,35,1)
def generator(x, y_label, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        cat1 = tf.concat([x, y_label], 3)
        print(cat1.shape)
        print('x:')
        print(x.shape)
        print('y_label:')
        print(y_label.shape)
        print('cat1')
        print(cat1.shape)
        # ?,28,28,101
        # 1st hidden layer
        deconv1 = tf.layers.conv2d_transpose(cat1, 34, [5, 15], strides=(2, 2), padding='valid', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)
        print('relu1')
        print(lrelu1.shape)
        # ?,34,34,256
        # 2nd hidden layer
        pool1 = tf.layers.max_pooling2d(inputs=lrelu1, pool_size=[2, 1], strides=2)

        deconv2 = tf.layers.conv2d_transpose(pool1, 32, [5, 15], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)
        print('relu1')
        print(lrelu2.shape)

        pool2 = tf.layers.max_pooling2d(inputs=lrelu2, pool_size=[2, 1], strides=2)
        # output layer
        deconv3 = tf.layers.conv2d_transpose(pool2, 1, [7, 15], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        print('relu1')
        print(deconv3.shape)
        o = tf.nn.tanh(deconv3)
        #
        return o

# D(x)
def discriminator(x, y_fill, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        cat1 = tf.concat([x, y_fill], 3)
        print('cat1')
        print(cat1.shape)
        # 1st hidden layer
        conv1 = tf.layers.conv2d(cat1, 34, [2, 8], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(conv1, 0.2)
        print('conv1')
        print(conv1.shape)
        # pool1 = tf.layers.max_pooling2d(inputs=lrelu1, pool_size=[2, 1], strides=2)
        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 32, [2, 8], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        print('conv2')
        print(conv2.shape)
        # pool2 = tf.layers.max_pooling2d(inputs=lrelu2, pool_size=[2, 1], strides=2)
        # output layer
        conv3 = tf.layers.conv2d(lrelu2, 1, [2, 8], strides=(1, 1), padding='valid', kernel_initializer=w_init)
        o = tf.nn.sigmoid(conv3)
        print('O - discriminator')
        print(o.shape)
        return o, conv3

def one_hot_enc(dataframe, _to_idx):
    vector = np.zeros((len(dataframe), dataframe.str.len().max(), len(_to_idx),1))
    for idx,x in enumerate(dataframe):
        indices = []
        for l in x:
            indices.append(_to_idx[l])
        vector[idx, np.arange(len(x)), indices] = 1
    return vector

# def string_vectorizer(str_input, alphabet):
#     vector = [[[0] if char != symbol else [1] for char in alphabet]
#                   for symbol in str_input]
#     return array(vector)

def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, y_label: fixed_y_, isTrain: False})

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (img_size, img_size)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 100
# lr = 0.0002
train_epoch = 30
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.0002, global_step, 500, 0.95, staircase=True)

train_words = []
train_ipa = []
fields = ['word', 'ipa']
train_df = pandas.read_csv('data/non-name-words.txt', encoding='utf-16-le', usecols=fields)
test_df = pandas.read_csv('data/name-words.txt', encoding='utf-16-le', usecols=fields)

chars = list(set("".join(train_df.word)))
idx_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_idx = {char:ix for ix, char in enumerate(chars)}

chars = list(set("".join(train_df.ipa)))
idx_to_ipa = {ix:char for ix, char in enumerate(chars)}
ipa_to_idx = {char:ix for ix, char in enumerate(chars)}

print("Character Vocab")
print (idx_to_char)
print("IPA Vocab")
print (idx_to_ipa)

x_train = one_hot_enc(train_df.ipa, ipa_to_idx)
y_train = one_hot_enc(train_df.word, char_to_idx)
x_test = one_hot_enc(test_df.ipa, ipa_to_idx)
y_test = one_hot_enc(test_df.word, char_to_idx)
print(x_train.shape)

# x shape:       (?,28,28,1)
# z shape:       (?,28,28,1)
# y_label shape: (?,29,35,1)
# y_fill shape:  (?,29,35,1)

# import tensorflow as tf
# import numpy as np

# t1 = tf.placeholder(tf.float32, [None, 400])
# t2 = tf.placeholder(tf.float32, [None, 1176])
# t3 = tf.concat([t1, t2], axis = 1)

# with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    t3_val = sess.run(t3, feed_dict = {t1: np.ones((300, 400)), t2: np.ones((300, 1176))})

#     print(t3_val.shape)
#      (300, 1576)

x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1], x_train.shape[2], 1))
z = tf.placeholder(tf.float32, shape=(None, x_train.shape[1], x_train.shape[2], 1))
y_label = tf.placeholder(tf.float32, shape=(None, y_train.shape[1], y_train.shape[2], 1))
y_fill = tf.placeholder(tf.float32, shape=(None, y_train.shape[1], y_train.shape[2], 1))
isTrain = tf.placeholder(dtype=tf.bool)

print('x shape')
print(x.shape)
print('z shape')
print(z.shape)

# x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 1))
# z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
# y_label = tf.placeholder(tf.float32, shape=(None, 1, 1, 10))
# y_fill = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 10))
# isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, y_label, isTrain)
print(G_z.shape)

D_real, D_real_logits = discriminator(x, y_fill, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y_fill, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, y_train.shape[1], y_train.shape[1], 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, y_train.shape[1], y_train.shape[1], 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, y_train.shape[1], y_train.shape[1], 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optim = tf.train.AdamOptimizer(lr, beta1=0.5)
    D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
    # D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_set = x_train

train_label = y_train


root = 'IPA_cDCGAN_results/'
model = 'IPA_cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    shuffle_idxs = random.sample(range(0, len(train_set)), len(train_set))
    shuffled_set = train_set[shuffle_idxs]
    shuffled_label = train_label[shuffle_idxs]
    for iter in range(len(shuffled_set) // batch_size):
        # update discriminator
        x_ = shuffled_set[iter*batch_size:(iter+1)*batch_size]
        print(shuffled_label[iter*batch_size:(iter+1)*batch_size].shape)
        y_label_ = shuffled_label[iter*batch_size:(iter+1)*batch_size].reshape([batch_size, y_train.shape[1], y_train.shape[2], 1])
        y_fill_ = y_label_ * np.ones([batch_size, y_train.shape[1], y_train.shape[2], 1])
        z_ = np.random.normal(0, 1, (batch_size, x_train.shape[1], x_train.shape[2], 1))

        print('x')
        print(x.shape)
        print(x_.shape)
        print('z')
        print(z.shape)
        print(z_.shape)
        print('y_fill')
        print(y_fill.shape)
        print(y_fill_.shape)
        print('y_label')
        print(y_label.shape)
        print(y_label_.shape)
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, y_fill: y_fill_, y_label: y_label_, isTrain: True})

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, y_train.shape[1], y_train.shape[1], 100))
        y_ = np.random.randint(0, 9, (batch_size, 1))
        y_label_ = onehot[y_.astype(np.int32)].reshape([batch_size, y_train.shape[1], y_train.shape[1], 1])
        y_fill_ = y_label_ * np.ones([batch_size, x_train.shape[1], x_train.shape[2], 1])
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y_fill: y_fill_, y_label: y_label_, isTrain: True})

        errD_fake = D_loss_fake.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        errD_real = D_loss_real.eval({x: x_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        errG = G_loss.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})

        D_losses.append(errD_fake + errD_real)
        G_losses.append(errG)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

# show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

# images = []
# for e in range(train_epoch):
#     img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
