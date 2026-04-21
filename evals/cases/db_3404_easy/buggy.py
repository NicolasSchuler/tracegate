from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        d = [0] * (3)
        d[0] = 0
        d[1] = d[2] = 1
        for i in range(3, n + 1):
            d.append(sum(d[1:]))
            d.pop(0)
        return d[2]
