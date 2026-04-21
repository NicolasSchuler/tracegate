from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def getMaximumConsecutive(self, coins: List[int]) -> int:
        ans = 1  # next value we want to make

        for coin in sorted(coins):
            if coin > ans:
                return ans
            ans += coin

    return ans
