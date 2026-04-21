from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        cnt = 0
        left = nums[0]
        right = sum(nums[0:])
        if left >= right:
            cnt += 1
        for i in range(1, len(nums) - 1):
            left += nums[i]
            right -= nums[i]
            if left >= right:
                cnt += 1
        return cnt
