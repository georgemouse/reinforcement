# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    for times in range(iterations):
        V = self.values.copy()
        for state in mdp.getStates():
            action_values = util.Counter()
            for action in mdp.getPossibleActions(state):
                for trans_state, prob in mdp.getTransitionStatesAndProbs(state, action):
                    action_values[action] += prob * (mdp.getReward( state, action, trans_state) + discount * V[trans_state])
            self.values[state] = action_values[action_values.argMax()]
            
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    " Q = R + gamma*SIGMA(P*V)"
    Q_value = self.values[state]
    for trans_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
        Q_value += self.discount * prob * self.values[trans_state]
        
    return Q_value

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    "reflect the next k+1 rewards and decide"
    possible_actions = self.mdp.getPossibleActions(state)
    if not possible_actions: #terminal state
        return None
    
    action_values = util.Counter()
    for action in possible_actions:
        for trans_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            action_values[action] += prob * (self.mdp.getReward( state, action, trans_state) + self.discount * self.values[trans_state])
            
    return action_values.argMax()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
