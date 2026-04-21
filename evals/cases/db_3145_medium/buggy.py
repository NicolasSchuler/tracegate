from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minIncrements(self, n: int, cost: List[int]) -> int:
        ans = 0
        n //= 2

        for i in reversed(range(n)):
            mn, mx = sorted((cost[2 * i + 1], cost[2 * i]))

            ans += mx - mn
            cost[i] += mx

        return ans
