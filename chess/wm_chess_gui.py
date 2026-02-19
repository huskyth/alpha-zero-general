import threading

import pygame
import os

from utils import dotdict

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import numpy as np

from chess.common import shiftOutChessman, DISTANCE, GAME_MAP
import copy

from chess.common import MOVE_TO_INDEX_DICT

BLACK = 1
WHITE = -1
SCREEN_WIDTH = 580
SCREEN_HEIGHT = 580
CHESSMAN_WIDTH = 20
CHESSMAN_HEIGHT = 20


class WMChessGUI:
    def __init__(self, mcts_player, play_state, human_color=-1, fps=6):

        # screen
        self.board = None
        self.width = 580
        self.height = 580

        self.fps = fps

        # human color
        self.human_color = human_color

        # reset items
        self.board = None
        self.number = None
        self.is_human = None
        self.human_move = None
        self.chessman_in_hand = None
        self.chosen_chessman_color = None
        self.chosen_chessman = None

        # reset status
        self.reset_status()

        self.mcts_player = mcts_player
        self.play_state = play_state

        self.current_player = 1

    def __del__(self):
        # close window
        self.is_running = False

    def init_point_status(self):
        self.board = []
        black = [0, 1, 2, 3, 4, 8]
        white = [7, 11, 12, 13, 14, 15]
        for x in range(21):
            self.board.append(0)
        for x in black:
            self.board[x] = BLACK
        for x in white:
            self.board[x] = WHITE
        self.board = np.array(self.board)

    # reset status
    def reset_status(self):
        self.init_point_status()

        self.is_human = False
        self.human_move = -1

        self.chessman_in_hand = False
        self.chosen_chessman_color = None
        self.chosen_chessman = None

    # human play
    def set_is_human(self, value=True):
        self.is_human = value

    def get_is_human(self):
        return self.is_human

    def get_human_move(self):
        return self.human_move

    def get_human_color(self):
        return self.human_color

    # execute move
    def execute_move(self, color, move, info=None):
        from_int, to_int = move
        print(f"🌿 exec {from_int} to {to_int}")
        assert color == WHITE or color == BLACK
        assert self.board[from_int] == color, f"{self.board[from_int]}, {color}, {self.board}"
        assert self.board[to_int] == 0
        assert DISTANCE[from_int][to_int] == 1
        self.board[from_int] = 0
        self.board[to_int] = color
        bake_point_status = copy.deepcopy(self.board)
        self.board = shiftOutChessman(
            bake_point_status, DISTANCE)

    def start(self):
        if os.name == 'posix':
            self.loop()
        else:
            t = threading.Thread(target=self.loop)
            t.start()

    # main loop
    def loop(self):
        # set running
        self.is_running = True

        # init
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("WmChess")

        # timer
        self.clock = pygame.time.Clock()

        # background image
        base_folder = os.path.dirname(__file__)
        self.background_img = pygame.image.load(
            os.path.join(base_folder, 'assets/watermelon.png')).convert()

        # font
        self.font = pygame.font.SysFont('Arial', 16)

        while self.is_running:
            # timer
            self.clock.tick(self.fps)

            # handle event
            for event in pygame.event.get():
                # close window
                if event.type == pygame.QUIT:
                    self.is_running = False

                # human play
                if self.is_human:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        print("🌿 Mouse button down")
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        chessman = self._chosen_chessman(mouse_x, mouse_y)
                        if chessman is None:
                            continue
                        if not self.chessman_in_hand:
                            if self.board[chessman] == self.human_color:
                                self.chosen_chessman_color = self.board[chessman]
                                self.chessman_in_hand = True
                                self.chosen_chessman = chessman
                                print(f"🌿 chessman in hand human_color is {self.human_color}")

                        else:
                            if self.board[chessman] == 0 and \
                                    DISTANCE[self.chosen_chessman][chessman] == 1:

                                self.human_move = (self.chosen_chessman, chessman)
                                self.human_move = MOVE_TO_INDEX_DICT[self.human_move]
                                self.set_is_human(False)
                                self.board, self.current_player = self.play_state.getNextState(self.board,
                                                                                               self.current_player,
                                                                                               self.human_move)
                            else:
                                self.board[
                                    self.chosen_chessman] = self.chosen_chessman_color
                            self.chessman_in_hand = False
                            # draw

                else:
                    x = self.play_state.getCanonicalForm(self.board, self.current_player)
                    action = np.argmax(self.mcts_player.getActionProb(x, temp=0))
                    self.board, self.current_player = self.play_state.getNextState(self.board, self.current_player,
                                                                                   action)
                    self.is_human = True
                    self.mcts_player.print(x)

                # draw
                self._draw_background()
                self._draw_chessman()

            # refresh
            pygame.display.flip()

    def _chosen_chessman(self, x, y):
        x, y = x / (SCREEN_WIDTH + 0.0), y / (SCREEN_HEIGHT + 0.0)
        for point in range(21):
            if abs(x - GAME_MAP[point][0]) < 0.05 and abs(y - GAME_MAP[point][1]) < 0.05:
                print(f"🌿 choose {point}")
                return point
        return None

    def _draw_background(self):
        # load background
        self.screen.blit(self.background_img, (0, 0))

    @staticmethod
    def fix_xy(target):
        x = GAME_MAP[target][0] * \
            SCREEN_WIDTH - CHESSMAN_WIDTH * 0.5
        y = GAME_MAP[target][1] * \
            SCREEN_HEIGHT - CHESSMAN_HEIGHT * 1
        return x, y

    def draw_end_string(self, string):
        text_surface = self.font.render(string, True, (255, 255, 0))
        self.screen.blit(text_surface, (500, 560))
        pygame.display.update()

    def _draw_chessman(self):
        for index, point in enumerate(self.board):
            if point == 0:
                continue
            (x, y) = WMChessGUI.fix_xy(index)
            color = [0, 0, 0] if point == BLACK else [255, 0, 0]
            if self.chessman_in_hand:
                if index == self.chosen_chessman:
                    color[1] += 126
            color = tuple(color)

            if point == BLACK:
                pygame.draw.circle(self.screen, color, (int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                                   int(CHESSMAN_HEIGHT // 2 * 1.5))
            elif point == WHITE:
                pygame.draw.circle(self.screen, color,
                                   (int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                                   int(CHESSMAN_HEIGHT // 2 * 1.5))


if __name__ == '__main__':
    from othello.pytorch.NNet import NNetWrapper as NNet
    from chess.chess_game import Chess as Game
    from MCTS import MCTS

    g = Game()
    n1 = NNet(g)
    n1.load_checkpoint('/Users/tenghao/Desktop/alpha-zero-general/temp', 'best.pth.tar')
    args1 = dotdict({'numMCTSSims': 800, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)
    wm = WMChessGUI(mcts1, g)
    wm.start()
