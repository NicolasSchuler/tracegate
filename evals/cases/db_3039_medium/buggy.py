from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        j = 0
        for i in range(1, n + 1):
            if n % i == 0:
                num = i
                j += 1
            if j == k:
                break
        return num if j == k + 1 else -1
