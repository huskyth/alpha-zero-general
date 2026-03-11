import logging

import coloredlogs

from Coach import Coach
from chess.chess_game import Chess as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 350,  #
    'updateThreshold': 0.6,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 100,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.8,

    'checkpoint': './temp_sim100_1.8_500_maxstep/',
    'load_model': False,
    'load_folder_file': ('./temp_sim100_1.8_500_maxstep', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'epsilon': 0.0,
    'dirAlpha': 0.0,

})

import swanlab

swanlab.login(api_key="rdGaOSnlBY0KBDnNdkzja")
swanlab = swanlab.init(project="Chess", logdir="logs")


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, swanlab)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args, swanlab)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
