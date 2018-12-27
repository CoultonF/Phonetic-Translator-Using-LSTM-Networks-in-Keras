from __future__ import print_function
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os, time, itertools, imageio, pickle, random, string
import tensorflow as tf
import keras as keras
from pandas import *
from numpy import array
from keras.models import Model
from keras.layers import Input, LSTM, Dense

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

def from_one_hot(onehot, _to_idx):
    onehot_as_string = []
    for idx,x in enumerate(onehot):
        onehot_as_string.append([])
        for l in x:
            l = l.flatten()

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = './data/1-11-letter-name-words-dataset.txt'
weights_file = 'Model/11-word-length-model.h5'
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
        # print(t," ",i," ",input_token_index[char]," ", char)
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.load_weights(weights_file)
# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
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

print('NAME LENGTH: ' + str(max_encoder_seq_length),file=open('output.txt', 'a'))
for seq_index, ss in enumerate(input_texts):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    # print(str(len(input_texts[seq_index])))
    # model.load_weights('Model/' + str(len(input_texts[seq_index])) + '-word-length-model.h5')
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    b=',\n'
    for char in b: target_texts[seq_index]=target_texts[seq_index].replace(char,"")
    print('Input Word:', input_texts[seq_index],'Input IPA', target_texts[seq_index],'Generated IPA:', decoded_sentence, file=open('output.txt', 'a'),end='')
    print('Input Word:', input_texts[seq_index],'Input IPA', target_texts[seq_index],'Generated IPA:', decoded_sentence,end='')
