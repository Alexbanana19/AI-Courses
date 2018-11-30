#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:39:53 2017

@author: alex
"""
import gym

ENV_NAME = 'Breakout-v0'

def main():
    env = gym.make(ENV_NAME)
    for i in xrange(1):
        env.render()
        action = 5
        print action
        next_state, r, done, _ = env.step(action)
        print next_state.shape
        

if __name__ == '__main__':
    main()