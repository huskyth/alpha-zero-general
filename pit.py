import Arena
from MCTS import MCTS
from chess.chess_game import Chess as Game
from othello.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

g = Game()
# nnet players

n1 = NNet(g)
n1.load_checkpoint('./temp', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNet(g)
# n2.load_checkpoint('./temp', 'best.pth.tar')
args2 = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

player2 = n2p

arena = Arena.Arena(n1p, player2, g)

print(arena.playGames(20, verbose=False))
