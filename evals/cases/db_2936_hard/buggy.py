from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        MOD = 10**9 + 7

        top_k_speeds = []
        res, top_k_sum = 0, 0

        pairs = [(e, s) for e, s in zip(efficiency, speed)]
        for e, s in sorted(pairs, reverse=True):
            heapq.heappush(top_k_speeds, s)
            top_k_sum += s

            if len(top_k_speeds) > k:
                top_k_sum -= heapq.heappop(top_k_speeds)

            res = max(res, e * top_k_sum)

        console.log(res % MOD)
        return res % MOD
