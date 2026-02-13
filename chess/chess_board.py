import copy
import numpy as np
from chess.common import GAME_MAP, LENGTH_OF_BOARD, BLACK, WHITE, DISTANCE, get_neighbours, shiftOutChessman, \
    INDEX_TO_MOVE_DICT


class ChessBoard:

    def __init__(self):
        self.gameMap = []
        self.pointStatus = []
        self.distance = []
        self.init_distance()
        self.init_point_status()
        self.init_game_map()
        self.turn = 0

    def get_game_map(self):
        return self.gameMap

    def get_point_status(self):
        return self.pointStatus

    def get_distance(self):
        return self.distance

    def init_distance(self):
        self.distance = DISTANCE

    def init_point_status(self):
        pointStatus = []
        black = [0, 1, 2, 3, 4, 8]
        white = [7, 11, 12, 13, 14, 15]
        for x in range(LENGTH_OF_BOARD):
            pointStatus.append(0)
        for x in black:
            pointStatus[x] = BLACK
        for x in white:
            pointStatus[x] = WHITE
        return np.array(pointStatus)

    def init_game_map(self):
        self.gameMap = GAME_MAP

    def get_legal_moves(self, pointStatus, player):
        assert player in [WHITE, BLACK]
        legal_moves_list = []
        for from_point_idx, chessman in enumerate(pointStatus):
            if chessman != player:
                continue
            to_point_idx_list = get_neighbours(from_point_idx, self.distance)
            for to_point_idx in to_point_idx_list:
                to_point = pointStatus[to_point_idx]
                if to_point != 0:
                    continue
                legal_moves_list.append((from_point_idx, to_point_idx))
        return legal_moves_list

    def execute_move(self, move, player, pointStatus):
        self.turn += 1
        if isinstance(move, int):
            move = INDEX_TO_MOVE_DICT[move]

        if isinstance(move, np.int32):
            move = INDEX_TO_MOVE_DICT[int(move)]

        from_int, to_int = move
        assert player == WHITE or player == BLACK
        assert pointStatus[from_int] == player
        assert pointStatus[to_int] == 0
        assert self.distance[from_int][to_int] == 1
        pointStatus[from_int] = 0
        pointStatus[to_int] = player
        bake_point_status = copy.deepcopy(pointStatus)
        return shiftOutChessman(bake_point_status, self.distance)

    def check_winner(self, pointStatus, player, depth):

        black_num = 0
        white_num = 0
        for color in pointStatus:
            if color == BLACK:
                black_num += 1
            elif color == WHITE:
                white_num += 1

        if black_num < 3 or white_num < 3:
            if black_num < 3:
                return 1 if player == WHITE else -1
            else:
                return 1 if player == BLACK else -1

        if depth >= 200:
            if black_num == white_num:
                return 1e-5
            elif black_num > white_num:
                return 1 if player == BLACK else -1
            else:
                return 1 if player == WHITE else -1

        return 0
