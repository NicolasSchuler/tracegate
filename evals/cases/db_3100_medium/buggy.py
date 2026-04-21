from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def numWays(self, s: str) -> int:
        MOD = 10**9 + 7
        N = len(s)
        c1 = s.count("1")
        if not c1:
            return (((N - 1) * (N - 2)) // 2) % MOD
        if c1 % 3:
            return 0
        idx1, idx2, idx3, idx4 = 0, 0, 0, 0
        cnt = 0
        for i in range(N):
            if s[i] == "1":
                cnt += 1
            if cnt == c1 // 3:
                idx1 = i
                break
        for i in range(idx1 + 1, N):
            if s[i] == "0":
                idx2 = i
                break
        for i in range(N - 1, -1, -1):
            if s[i] == "1":
                cnt += 1
            if cnt == c1 // 3:
                idx4 = i
                break
        for i in range(idx4 - 1, -1, -1):
            if s[i] == "1":
                idx3 = i
                break
        return ((idx2 - idx1) * (idx4 - idx3)) % MOD
