import copy
from pathlib import Path
import cv2
import numpy as np

from chess.chess_board import ChessBoard
from chess.common import from_array_to_input, GAME_MAP, MOVE_TO_INDEX_DICT, INDEX_TO_MOVE_DICT
from chess.symmetry_creator import lr, tb_, LEFT_ACTION_INDEX, RIGHT_ACTION_INDEX, TOP_ACTION_INDEX, \
    BOTTOM_ACTION_INDEX

ROOT_PATH = Path(__file__).parent
SCREEN_WIDTH = 580
SCREEN_HEIGHT = 580
CHESSMAN_WIDTH = 20
CHESSMAN_HEIGHT = 20
BLACK = 1
WHITE = -1
MAX_DRAW_TIME = 3


class Chess(ChessBoard):
    def __init__(self, start_player=1, is_render=False):
        self.current_player = start_player
        super().__init__()
        self.move_to_index = MOVE_TO_INDEX_DICT
        self.index_to_move = INDEX_TO_MOVE_DICT
        self.is_render = is_render

    def is_end(self, mock=False):
        winner = self.check_winner(mock)
        is_end = winner is not None
        return is_end, winner

    @staticmethod
    def _fix_xy(target):
        x = GAME_MAP[target][0] * \
            SCREEN_WIDTH - CHESSMAN_WIDTH * 0.5
        y = GAME_MAP[target][1] * \
            SCREEN_HEIGHT - CHESSMAN_HEIGHT * 1
        return x, y

    def _write_point(self):
        image = cv2.imread(str(ROOT_PATH / "assets/watermelon.png"))
        for index, point in enumerate(self.pointStatus):
            if point == 0:
                continue
            (x, y) = Chess._fix_xy(index)
            if point == BLACK:
                cv2.circle(img=image, color=(0.0, 0.0, 0.0),
                           center=(int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                           radius=int(CHESSMAN_HEIGHT // 2 * 1.5), thickness=-1)
            elif point == WHITE:
                cv2.circle(img=image, color=(0.0, 0.0, 255.0),
                           center=(int(x + CHESSMAN_WIDTH / 2), int(y + CHESSMAN_HEIGHT / 2)),
                           radius=int(CHESSMAN_HEIGHT // 2 * 1.5), thickness=-1)
        return image

    def render(self, key):
        if not self.is_render:
            return
        print(f"当前局面{self.pointStatus}的日志如下\n{key}\n")

    def center_probability(self, pi):
        l, r = np.array(LEFT_ACTION_INDEX), np.array(RIGHT_ACTION_INDEX)
        new_pi = copy.deepcopy(pi)
        new_pi[l], new_pi[r] = new_pi[r], new_pi[l]
        t, b = np.array(TOP_ACTION_INDEX), np.array(BOTTOM_ACTION_INDEX)
        new_pi = copy.deepcopy(new_pi)
        new_pi[t], new_pi[b] = new_pi[b], new_pi[t]
        return new_pi

    def get_torch_state(self):
        """
            得到棋盘的张量
            :return:
        """
        state = from_array_to_input_tensor(self.pointStatus, self.current_player)
        return state

    def get_current_player(self):
        return self.current_player

    def reset(self, start_player=1):
        self.init_point_status()
        self.current_player = start_player
        self.turn = 0

    def move_random(self):
        import random
        l_move = self.get_legal_moves(self.get_current_player())
        l_move = random.choice(l_move)
        max_act = self.move_to_index[l_move]
        return max_act

    def top_buttom(self, s, p):
        new_board, new_pi = tb_(s, p)
        return new_board, new_pi.tolist()

    def image_show(self, key, is_image_show, wait_key=5):
        if not is_image_show:
            return
        img = self._write_point()
        cv2.imshow(key, img)
        return cv2.waitKey(wait_key)

    def left_right(self, s, p):
        new_board, new_pi = lr(s, p)
        return new_board, new_pi.tolist()

    def center(self, s, p):
        new_board, new_pi = lr(s, p)
        new_board, new_pi = tb_(new_board, new_pi)
        return new_board, new_pi.tolist()

    def stringRepresentation(self, board):
        return board.tobytes()

    def getActionSize(self):
        return 72

    def getGameEnded(self, board, player, depth):
        return self.check_winner(board, player, depth)

    def getValidMoves(self, board, player):
        valids = [0] * self.getActionSize()
        f_t_list = self.get_legal_moves(board, player)
        for item in f_t_list:
            idx = MOVE_TO_INDEX_DICT[item]
            valids[idx] = 1
        return np.array(valids)

    def getCanonicalForm(self, board, player):
        return player * board

    def getNextState(self, board, player, action):
        board_ = copy.deepcopy(board)
        ret_board = self.execute_move(action, player, board_)
        assert id(ret_board) != id(board)
        det = abs(board_[board_ == -player].sum().item()) - abs(ret_board[ret_board == -player].sum().item())
        return ret_board, -player, abs(det)

    def getInitBoard(self):
        return self.init_point_status()

    def getSymmetries(self, board, pi):
        board = from_array_to_input(board)
        pi = np.array(pi)
        l = [(board, pi.tolist())]
        nb, npi = self.left_right(board, pi)
        l += [(nb, npi)]
        nb, npi = self.top_buttom(board, pi)
        l += [(nb, npi)]
        nb, npi = self.center(board, pi)
        l += [(nb, npi)]
        return l

    def getBoardSize(self):
        # (a,b) tuple
        return 7, 7


if __name__ == '__main__':
    g = Chess()
    intb = g.getInitBoard()
    _, _, c = g.getNextState(intb, 1, 9)
    print(c)
