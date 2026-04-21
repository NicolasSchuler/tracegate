from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        nums = sorted(nums)
        min = float("inf")
        if len(nums) < 2:
            return 0
        for i in range(len(nums) - 1):
            x = abs(nums[i] - nums[i + 1])
            if min < x:
                min = x
        return min
