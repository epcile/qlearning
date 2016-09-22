#!/usr/bin/python
from numpy.random import uniform as uniform
from collections import namedtuple as namedtuple
import numpy

# (action, from, to, reward)
AUTOMATON = (('upC', 0, 2, 0),
 ('upC', 1, 3, 0),
 ('upC', 2, 4, 0),
 ('upC', 3, 5, 0),
 ('upC', 4, 4, -1),
 ('upC', 5, 5, -1),
 ('right', 0, 1, 0),
 ('right', 1, 1, -1),
 ('right', 2, 3, 0),
 ('right', 3, 3, -1),
 ('right', 4, 5, 0),
 ('right', 5, 5, -1),
 ('left', 0, 0, -1),
 ('left', 1, 0, 0),
 ('left', 2, 2, -100),
 ('left', 3, 2, 0),
 ('left', 4, 0, 10),
 ('left', 5, 4, 0))
 
class Environment(object):
  def __init__(self):
    self._state = 0

  @property
  def state(self):
    return self._state

  def trigger(self, action):
    """Returns the reward and updates the state."""
    action_idx = action
    if action == 3:
      prob = uniform()
      if prob < 0.8:
        action_idx = 0
      elif prob < 0.9:
        action_idx = 1
      else:
        action_idx = 2
    action_entry = AUTOMATON[action_idx * 6 + self._state]
    print action_entry
    self._state = action_entry[2]
    return self._state, action_entry[3]


Experience = namedtuple('experience', ['s0', 'a', 's1', 'r'])
class Agent(object):
  def __init__(self, discount_rate=1.0):
    self._Q = numpy.zeros((6, 4), dtype='float')
    self._dr = discount_rate

  @property
  def Q(self):
    return self._Q

  def learn(self, e):
    max_action = 0
    max_q = 0
    for i in range(4):
      if self._Q[e.s1][i] > max_q:
        max_q = self._Q[e.s1][i]
        max_action = i

    self._Q[e.s0][e.a] = max_q * self._dr + e.r

def train(e, agent, warmup_steps=1000, eval_steps=100):
  for i in range(warmup_steps):
    action = numpy.random.randint(0, 4)
    old_state = e.state
    new_state, reward = e.trigger(action)
    ex = Experience(s0=old_state, a=action, s1=new_state, r=reward)
    agent.learn(ex)
    print i, ex, agent.Q
  print agent.Q
  ar = 0
  for i in range(eval_steps):
    old_state = e.state
    best_action = numpy.argmax(agent.Q[old_state])
    _, reward = e.trigger(best_action)
    print 'ba=', best_action, ', old_state=', old_state
    ar += reward
     
  print 'rewards = ', ar

def main():
  numpy.random.seed(12345)
  e = Environment()
  agent = Agent()
  train(e, agent)

if __name__ == "__main__":
  main()
