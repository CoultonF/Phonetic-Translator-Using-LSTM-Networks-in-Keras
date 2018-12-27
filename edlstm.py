from __future__ import print_function
import pandas
import random
import numpy as np
import matplotlib.pyplot as plt
import os, time, itertools, imageio, pickle, random, string

import glob
from importlib import reload
from pandas import *
from numpy import array

import keras as keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping

def one_hot_enc_gen(dataframe, _to_idx, str_len_max):
    vector = np.zeros((len(dataframe),str_len_max, len(_to_idx)))
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
    vector = np.zeros((len(dataframe), dataframe.str.len().max(), len(_to_idx)))
    for idx,x in enumerate(dataframe):
        indices = []
        for l in x:
            indices.append(_to_idx[l])
        vector[idx, np.arange(len(x)), indices] = 1
        if len(indices) < dataframe.str.len().max():
            empty = np.where(~vector[idx].any(axis=1))[0]
            vector[idx, np.arange(empty[0], dataframe.str.len().max()), len(_to_idx)-1] = 1
    return vector

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index[',']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def from_one_hot(onehot, _to_idx):
    onehot_as_string = []
    for idx,x in enumerate(onehot):
        onehot_as_string.append([])
        for l in x:
            l = l.flatten()
            print(l)
for file in list(glob.glob('./data/*-letter-non-name-words-dataset.txt')):

    reload(keras)
    batch_size = 64  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 10000  # Number of samples to train on.
    # Path to the data txt file on disk.
    data_path = file

    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:

        input_text, target_text = line.split(',')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = ',' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = ['b', 'e', 't', 'm', 'x', 'r', 'v', 'j', 'a', 'i', 'w', 'n', 'y', 'p', 'g', 'c', 'l', 'o', 's', 'h', 'f', 'q', 'u', 'z', 'd', 'k']
    target_characters = [',','ð', 'ŋ', 'ɡ', 'b', 'ɫ', 'e', 't', 'm', 'ɝ', 'ɔ', 'v', 'j', 'ɹ', 'a', 'ɛ', 'i', 'w', 'ʃ', 'n', 'θ', 'ʒ', 'p', 'æ', 'o', 's', 'ɑ', 'h', 'f', 'ɪ', 'u', 'ʊ', 'z', 'd', 'k', 'ə', '\n']
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(target_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:

                decoder_target_data[i, t - 1, target_token_index[char]] = 1.


    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    tbCallBack = keras.callbacks.TensorBoard('logs/' + str(max_encoder_seq_length) + '-word-length/', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[tbCallBack, early_stopping])
    model.save('Model/' + str(max_encoder_seq_length) + '-word-length-model.h5')


    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())
    with open('output.txt', 'a+') as f:
        print("WORD LENGTH: ", str(max_encoder_seq_length),file=f)
    for seq_index in random.sample(range(1,len(input_texts)), 10):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_characters = decode_sequence(input_seq)
        b='\n,'
        for char in b: target_texts[seq_index]=target_texts[seq_index].replace(char,"")
        with open('output.txt', 'a+') as f:
            print('Input Word:', input_texts[seq_index],'Input IPA', target_texts[seq_index],'Generated IPA:', decoded_characters, file=f, end='')

    model.reset_states()
    keras.backend.clear_session()
    encoder_model.reset_states()
