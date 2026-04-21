from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def findValueOfPartition(self, nums: Array[int]) -> int:
        nums.sort()
        min_diff = float("inf")

        for i in range(1, len(nums)):
            min_diff = min(min_diff, abs(nums[i] - nums[i - 1]))

        return min_diff
