from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
        n = len(nums)
        total = sum(nums)
        s1 = 0
        s2 = total
        arr = []
        for i in range(n):
            arr.append(self.calculateAbsoluteDifference(s1, nums[i], i, s2, n))
            s1 = s1 + nums[i]
            s2 = total - s1

        return arr
