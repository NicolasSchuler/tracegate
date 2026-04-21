from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        ans = [-1] * len(nums)
        s, ss = [], []
        for i, x in enumerate(nums):
            while ss and nums[ss[-1]] < x:
                ans[ss.pop()] = x
            buff = []
            while s and nums[s[-1]] < x:
                buff.append(s.pop())
            while buff:
                s.append(buff.pop())
            s.append(i)
        return ans
