from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        def get_neighbors(board):
            neighbors = []
            r, c = 0, 0
            for i in range(2):
                for j in range(3):
                    if board[i][j] == 0:
                        r, c = i, j
            for i, j in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_r, new_c = r + i, c + j
                if 0 <= new_r < 2 and 0 <= new_c < 3:
                    new_board = [row[:] for row in board]
                    new_board[r][c] = new_board[new_r][new_c]
                    new_board[new_r][new_c] = 0
                    neighbors.append(new_board)
            return neighbors

        queue = deque()
        queue.append((board, 0))
        seen = set()
        seen.add(tuple(tuple(row) for row in board))

        while queue:
            board, moves = queue.popleft()
            if board == [[1, 2, 3], [5, 4, 0]]:
                return moves
            for neighbor in get_neighbors(board):
                if tuple(tuple(row) for row in neighbor) not in seen:
                    queue.append((neighbor, moves + 1))
                    seen.add(tuple(tuple(row) for row in neighbor))
        return -1
