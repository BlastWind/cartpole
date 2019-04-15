# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:57:32 2019

@author: andre
"""


import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make("CartPole-v1")

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n


LEARNING_RATE = 0.001

def RL_model(observation_space, action_space):
        Model = Sequential()
        Model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        Model.add(Dense(24, activation="relu"))
        Model.add(Dense(action_space, activation="linear"))
        Model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        Model.summary()
        
        return Model
    
def act(model, state):
        q_values = model.predict(state)
        return np.argmax(q_values[0])
    
myModel = RL_model(observation_space, action_space)
myModel.load_weights('./weights/model_weights_at_step100.h5')

def test():
    for _ in range(3):
        
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        
        while True:
                step += 1
                env.render()
                action = act(myModel, state)
                state_next,reward,terminal,info = env.step(action)
                
                state_next = np.reshape(state_next, [1, observation_space])
                state = state_next
                
                
                if terminal:
                    print("score: " + str(step))
                    
                    break
               
        env.close()
        
if __name__ == "__main__":
    test()