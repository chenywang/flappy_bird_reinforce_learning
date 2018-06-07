#!/usr/bin/env python
from __future__ import print_function

import sys

import cv2
import tensorflow as tf

sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

TRAIN = True
OUTPUT_GRAPH = False
GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations

if not TRAIN:
    OBSERVE = 100000.  # timesteps to observe before training
    EXPLORE = 2000000.  # frames over which to anneal epsilon
    FINAL_EPSILON = 0.0001  # final value of epsilon
    INITIAL_EPSILON = 0.0001  # starting value of epsilon
else:
    # OBSERVE = 10000
    OBSERVE = 100
    EXPLORE = 3000000
    FINAL_EPSILON = 0.0001
    INITIAL_EPSILON = 0.1

REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    input = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(input, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    output = tf.matmul(h_fc1, W_fc2) + b_fc2

    return input, output


def trainNetwork(input, output, sess):
    # define the cost function
    action_place_holder = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(output, action_place_holder), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up action_place_holder game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    memory = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    init_page_colored, init_reward, terminal = game_state.frame_step(do_nothing)
    init_page_grey = cv2.cvtColor(cv2.resize(init_page_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, init_page_binary = cv2.threshold(init_page_grey, 1, 255, cv2.THRESH_BINARY)
    four_pages_from_memory = np.stack((init_page_binary, init_page_binary, init_page_binary, init_page_binary), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if not TRAIN:
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    step = 0
    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)
    while True:
        # choose an action epsilon greedily
        network_output = output.eval(feed_dict={input: [four_pages_from_memory]})[0]
        action = np.zeros([ACTIONS])
        action_index = 0
        if step % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                action[action_index] = 1
            else:
                action_index = np.argmax(network_output)
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and step > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        next_page_colored, current_reward, terminal = game_state.frame_step(action)
        next_page_grey = cv2.cvtColor(cv2.resize(next_page_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, next_page_binary = cv2.threshold(next_page_grey, 1, 255, cv2.THRESH_BINARY)
        next_page_binary = np.reshape(next_page_binary, (80, 80, 1))
        next_four_page = np.append(next_page_binary, four_pages_from_memory[:, :, :3], axis=2)

        # store the transition in memory
        memory.append((four_pages_from_memory, action, current_reward, next_four_page, terminal))
        if len(memory) > REPLAY_MEMORY:
            memory.popleft()

        # only train if done observing
        if step > OBSERVE:
            # sample action_place_holder minibatch to train on
            case_batch = random.sample(memory, BATCH)

            # get the case_batch variables
            four_pages_from_memory = [d[0] for d in case_batch]
            action_from_memory = [d[1] for d in case_batch]
            current_reward_from_memory = [d[2] for d in case_batch]
            terminal_from_memory = [d[3] for d in case_batch]

            y_batch = []
            network_output_from_memory = output.eval(feed_dict={input: four_pages_from_memory})
            for i, case in enumerate(case_batch):
                terminal = case[4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(current_reward_from_memory[i])
                else:
                    y_batch.append(current_reward_from_memory[i] + GAMMA * np.max(network_output_from_memory[i]))

            # perform gradient step
            train_step.run(feed_dict={
                y: y_batch,
                action_place_holder: action_from_memory,
                input: four_pages_from_memory}
            )
            # print("cost", cost.eval(feed_dict={
            #     y: y_batch,
            #     action_place_holder: action_from_memory,
            #     input: four_pages_from_memory}
            # ))

        # update the old values
        four_pages_from_memory = next_four_page
        step += 1

        # save progress every 10000 iterations
        if step % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=step)

        # print info
        if step <= OBSERVE:
            state = "observe"
        elif step <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", step, "/ STATE", state, \
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", current_reward, \
              "/ Q_MAX %e" % np.max(network_output))


def playGame():
    sess = tf.InteractiveSession()
    input, output = createNetwork()
    trainNetwork(input, output, sess)


def main():
    playGame()


if __name__ == "__main__":
    main()
