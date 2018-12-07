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
# z shape:       (?,9,27,1)
# y_label shape: (?,9,36,1)
def generator(x, y_label, isTrain=True, reuse=False):
    print("Generator")
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)
        print (x.shape)
        print (y_label.shape)
        # concat layer
        cat1 = tf.concat([x, y_label], 1)
        print("CONCAT")
        print(cat1.shape)
        # ?,18,27,1
        # 1st hidden layer
        deconv1 = tf.layers.conv2d_transpose(cat1, 256, [2, 1], strides=(1, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        # ?,19,54,256
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)
        pool1 = tf.layers.max_pooling2d(inputs=lrelu1, pool_size=[2, 3], strides=(2, 3))
        # ?,9,18,256

        print('CONV1')
        print(lrelu1.shape)
        print(pool1.shape)
        # 2nd hidden layer ?better to have 1D stride as 2 in deconv1, deconv2 or deconv3?
        deconv2 = tf.layers.conv2d_transpose(pool1, 128, [2, 1], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        # ?,19,36,256
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)
        pool2 = tf.layers.max_pooling2d(inputs=lrelu2, pool_size=[2, 2], strides=(2, 2))
        print('CONV2')
        print(lrelu2.shape)
        print(pool2.shape)
        # output layer ?better to have 1D window size as 2 in deconv1, deconv2, or dconv3(too small by 0.5 in pooling layer)?
        deconv3 = tf.layers.conv2d_transpose(pool2, 1, [2, 1], strides=(1, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        o = tf.nn.tanh(deconv3)
        print('CONV3')
        print (o.shape)
        return o

# D(x)
def discriminator(x, y_fill, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        cat1 = tf.concat([x, y_fill], 3)
        print("CONCAT")
        print(cat1.shape)

        # 1st hidden layer
        conv1 = tf.layers.conv2d(cat1, 128, [1, 8], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(conv1, 0.2)
        print('CONV1')
        print(lrelu1.shape)
        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [1, 9], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        print('CONV2')
        print(lrelu2.shape)
        # output layer
        conv3 = tf.layers.conv2d(lrelu2, 1, [1, 10], strides=(1, 1), padding='valid', kernel_initializer=w_init)
        o = tf.nn.sigmoid(conv3)
        print('CONV3')
        print (o.shape)
        return o, conv3
def one_hot_enc_gen(dataframe, _to_idx, str_len_max):
    vector = np.zeros((len(dataframe),str_len_max, len(_to_idx),1))
    for idx,x in enumerate(dataframe):
        indices = []
        for l in x:
            indices.append(_to_idx[l])
        vector[idx, np.arange(len(x)), indices] = 1
        if len(indices) < str_len_max:
            empty = np.where(~vector[idx].any(axis=1))[0]
            vector[idx, np.arange(empty[0], str_len_max), len(_to_idx)-1] = 1
    return vector
# dataframe.str.len().max()
def one_hot_enc(dataframe, _to_idx):
    vector = np.zeros((len(dataframe), dataframe.str.len().max(), len(_to_idx),1))
    for idx,x in enumerate(dataframe):
        indices = []
        for l in x:
            indices.append(_to_idx[l])
        vector[idx, np.arange(len(x)), indices] = 1
        if len(indices) < dataframe.str.len().max():
            empty = np.where(~vector[idx].any(axis=1))[0]
            vector[idx, np.arange(empty[0], dataframe.str.len().max()), len(_to_idx)-1] = 1
    return vector

def from_one_hot(onehot, _to_idx):
    onehot_as_string = []
    for idx,x in enumerate(onehot):
        onehot_as_string.append([])
        for l in x:
            l = l.flatten()
            print(l)
            # onehot_as_string[idx].append(_to_idx[l])

# def string_vectorizer(str_input, alphabet):
#     vector = [[[0] if char != symbol else [1] for char in alphabet]
#                   for symbol in str_input]
#     return array(vector)

def show_result(num_epoch, epoch_label, show = False, save = False, path = 'result.png', ):
    # print('G_z')
    # print(G_z.shape)
    # print('z')
    # print(z.shape)
    # print('z_fixed_')
    # print(fixed_z_.shape)
    # print('y_label')
    # print(y_label.shape)
    # print('fixed_y_')
    # print(fixed_y_.shape)
    test_ipa = sess.run(G_z, {z: fixed_z_, y_label: fixed_y_, isTrain: False})
    word_str=''
    ipa_str=''
    with open('results.txt', 'a+') as f, open('raw-data.txt', 'a+') as r:
        print(epoch_label, file=f)
        print(epoch_label, file=r)
        for idx,word in enumerate(test_ipa):
            word_str=''
            ipa_str=''
            raw_ipa_onehot= np.array([])
            for letter in y_label_[idx]:
                word_str += idx_to_char[np.argmax(letter)]
            # print(word_str)
            for letter in word:
                raw_ipa_onehot = word.reshape(9,36)
                ipa_str += idx_to_ipa[np.argmax(letter)]
            print('Word:',word_str,'\Real IPA:',generative_df.ipa[idx],'\tGen. IPA:',raw_ipa_onehot.flatten(), file=r)
            print('Word:',word_str,'\tGen. IPA:',ipa_str,'\tReal IPA:',generative_df.ipa[idx], file=f)

    # size_figure_grid = 10
    # fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    # for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    #     ax[i, j].get_xaxis().set_visible(False)
    #     ax[i, j].get_yaxis().set_visible(False)
    #
    # for k in range(10*10):
    #     i = k // 10
    #     j = k % 10
    #     ax[i, j].cla()
    #     ax[i, j].imshow(np.reshape(test_images[k], (img_size, img_size)), cmap='gray')
    #
    # label = 'Epoch {0}'.format(num_epoch)
    # fig.text(0.5, 0.04, label, ha='center')
    #
    # if save:
    #     plt.savefig(path)
    #
    # if show:
    #     plt.show()
    # else:
    #     plt.close()

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

train_epoch = 100
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.0002, global_step, 500, 0.95, staircase=True)

train_words = []
train_ipa = []
fields = ['word', 'ipa']
train_df = pandas.read_csv('data/10000-5-letter-non-name-words-dataset.txt', encoding='utf-16-le', usecols=fields)
generative_df = pandas.read_csv('data/100-5-letter-non-name-words-dataset.txt', encoding='utf-16-le', usecols=fields)

chars = list(set("".join(train_df.word)))
idx_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_idx = {char:ix for ix, char in enumerate(chars)}
idx_to_char[len(idx_to_char)] = ' '
char_to_idx[' '] = len(char_to_idx)

chars = list(set("".join(train_df.ipa)))
idx_to_ipa = {ix:char for ix, char in enumerate(chars)}
ipa_to_idx = {char:ix for ix, char in enumerate(chars)}
idx_to_ipa[len(idx_to_ipa)] = ' '
ipa_to_idx[' '] = len(ipa_to_idx)

onehot_df = pandas.DataFrame(columns=list("w"))
onehot_df.loc[0] = ["adrienne"]


print("Character Vocab")
print (idx_to_char)
print("IPA Vocab")
print (idx_to_ipa)

x_train = one_hot_enc(train_df.ipa, ipa_to_idx)
y_train = one_hot_enc(train_df.word, char_to_idx)

onehot = one_hot_enc_gen(generative_df.word, char_to_idx, y_train.shape[1])


# x shape:       (?,28,28,1)
# z shape:       (?,28,28,1)
# y_label shape: (?,29,35,1)
# y_fill shape:  (?,29,35,1)
temp_z_ = np.eye(y_train.shape[2])[np.random.choice(y_train.shape[2],y_train.shape[1])].reshape(1,y_train.shape[1],y_train.shape[2],1)
fixed_z_ = temp_z_
for i in range(batch_size - 1):
    fixed_z_ = np.concatenate([fixed_z_, temp_z_],0)
fixed_y_ = onehot.reshape(100, y_train.shape[1], y_train.shape[2], 1)

x = tf.placeholder(tf.float32, shape=(None, x_train.shape[1], x_train.shape[2], 1))
z = tf.placeholder(tf.float32, shape=(None, y_train.shape[1], y_train.shape[2], 1))
y_label = tf.placeholder(tf.float32, shape=(None, y_train.shape[1], y_train.shape[2], 1))
y_fill = tf.placeholder(tf.float32, shape=(None, x_train.shape[1], x_train.shape[2], 1))
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
print('discriminator real')
D_real, D_real_logits = discriminator(x, y_fill, isTrain)
print('discriminator fake')
D_fake, D_fake_logits = discriminator(G_z, y_fill, isTrain, reuse=True)
print("discriminator")
print(D_real.shape)
print(D_fake.shape)
# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, y_train.shape[1], y_train.shape[2], 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, y_train.shape[1], y_train.shape[2], 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, y_train.shape[1], y_train.shape[2], 1])))

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
        # print(shuffled_label.shape)
        # print(shuffled_set.shape)
        y_label_ = shuffled_label[iter*batch_size:(iter+1)*batch_size]
        # print("YLABEL")

        y_fill_ = np.ones([batch_size, x_train.shape[1], x_train.shape[2], 1])
        z_ = np.random.normal(0, 1, (batch_size, y_train.shape[1], y_train.shape[2], 1))
        # #
        # print('x')
        # print(x.shape)
        # print(x_.shape)
        # print('z')
        # print(z.shape)
        # print(z_.shape)
        # print('y_fill')
        # print(y_fill.shape)
        # print(y_fill_.shape)
        # print('y_label')
        # print(y_label.shape)
        # print(y_label_.shape)
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, y_fill: y_fill_, y_label: y_label_, isTrain: True})
        # print(loss_d_)
        # update generator
        z_ = np.random.normal(0, 1, (batch_size, y_train.shape[1], y_train.shape[2], 1))
        y_ = np.random.randint(0, 1, (batch_size, y_train.shape[1], y_train.shape[2], 1))
        # print(y_label_.shape)
        # for y_y in y_label_[0]:
        #     print(y_y.flatten())
        y_label_ = onehot.reshape([batch_size, y_train.shape[1], y_train.shape[2], 1])
        # print(y_label_.shape)
        y_fill_ = np.ones([batch_size, x_train.shape[1], x_train.shape[2], 1])
        # print('z_')
        # print(z.shape)
        # print(z_.shape)
        # print('y_label_')
        # print(y_label.shape)
        # print(y_label_.shape)
        # for y_y in y_label_[0]:
        #     print(y_y.flatten())
        # print('y_fill_')
        # print(y_fill.shape)
        # print(y_fill_.shape)
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y_fill: y_fill_, y_label: y_label_, isTrain: True})




        errD_fake = D_loss_fake.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        errD_real = D_loss_real.eval({x: x_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        errG = G_loss.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})

        D_losses.append(errD_fake + errD_real)
        G_losses.append(errG)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    printing_str = '[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses))
    print(printing_str)
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), printing_str, save=True, path=fixed_p)
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

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

# # images = []
# # for e in range(train_epoch):
# #     img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
# #     images.append(imageio.imread(img_name))
# # imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)
