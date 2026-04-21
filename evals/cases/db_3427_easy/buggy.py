from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minSubsequence(self, nums: List[int]) -> List[int]:
        total = sum(nums)
        nums.sort(reverse=True)
        sub_sum, sub_seq = 0, []
        for x in nums:
            sub_sum += x
            sub_seq.append(x)
            if sub_sum > total:
                return sub_seq
            sub_sum -= x
