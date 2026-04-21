from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def numPermsDISequence(self, s: str) -> int:
        mem = defaultdict(int)

        def dfs(i, val=0):
            if i == len(s):
                return 1
            if (i, val) in mem:
                return mem[i, val]
            p = 0
            if s[i] == "D":
                for ind in range(0, val + 1):
                    p += dfs(i + 1, ind) % (10**9 + 7)
            else:
                for ind in range(val + 1, i + 1):
                    p += dfs(i + 1, ind) % (10**9 + 7)
            mem[i, val] = p
            return p

        return dfs(0)
