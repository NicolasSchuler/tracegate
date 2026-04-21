from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        n = self.length(nums)
        low = 0
        high = n - 1
        if n == 1:
            return 0

        while low <= high:
            mid = (low + high) // 2
            if (mid == 0 or nums[mid] >= nums[mid - 1]) and (
                mid == n - 1 or nums[mid] >= nums[mid + 1]
            ):
                return mid
            elif nums[mid] <= nums[mid + 1]:
                low = mid + 1
            else:
                high = mid - 1
        return -1
