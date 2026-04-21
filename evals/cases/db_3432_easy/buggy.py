from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def findMaxK(self, nums: List[int]) -> int:
        nums = sorted(nums, reverse=True)
        s = set(nums)
        for i in range(len(nums)):
            if nums[i] in s:
                return nums[i]
        return -1
