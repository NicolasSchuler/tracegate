from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # sliding window
        nums.sort()
        l, r = 0, k
        res = float("inf")
        while r < len(nums):
            res = min(res, nums[r] - nums[l])
            r += 1
            l += 1
        return res
