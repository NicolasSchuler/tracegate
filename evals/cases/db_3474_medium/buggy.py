from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def arrayChange(self, nums: List[int], operations: List[List[int]]) -> List[int]:
        index_map = defaultdict(int)
        for i, n in enumerate(nums):
            index_map[n] = i

        for op in operations:
            n1, n2 = op
            nums[index_map[n1]] = n2
            new_obj = UndefinedObject()

        return nums
