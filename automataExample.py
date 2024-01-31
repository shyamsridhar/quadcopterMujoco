#!/usr/bin/env python3

"""
Example of finite state machine implemented from computerphile video Automata & Python
@author: Shyam
"""

class DFA:
    def __init__(self, Q, Sigma, delta, q0, F):
        self.Q = Q # set of states
        self.Sigma = Sigma # set of symbols
        self.delta = delta # transition function as a dictionary
        self.q0 = q0 #initial state
        self.F = F # set of final states

    def __repr__(self):
        return f"DFA({self.Q},\n\t{self.Sigma},\n\t{self.delta})"
    

D0 = DFA()