from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
        gc = 0
        for i in numsDivide:
            gc = int(math.gcd(gc, i))
        nums.sort()
        for i in range(len(nums)):
            if nums[i] % gc == 0:
                return i
        return -1
