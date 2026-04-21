from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums.sort()
        res = 100000

        for i in range(len(nums) - k):
            arr = nums[i : i + k]
            res = min(res, arr[-1] - arr[0])

        return res
