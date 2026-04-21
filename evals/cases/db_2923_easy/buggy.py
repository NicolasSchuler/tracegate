from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if numRows == 1:
            return [[1]]
        prev = self.generate(numRows - 1)
        fin = prev[-1]
        now = Array(1)
        for i in range(len(fin) - 1):
            now.append(fin[i] + fin[i + 1])
        now.push(1)
        prev.append(now)
        return prev
