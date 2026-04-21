from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def gridGame(self, grid: List[List[int]]) -> int:
        result = float("inf")
        left, right = 0, sum(grid[0])

        for a, b in zip(grid[1], grid[0]):
            right -= a

            result = min(result, max(left, right))
            left += b

        return result
