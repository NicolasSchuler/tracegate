from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minimumDeletions(self, nums: List[int]) -> int:
        nums = sorted(nums)
        return min(
            (
                min(nums.index(min(nums)) + 1, len(nums) - nums.index(min(nums)))
                + min(nums.index(max(nums)) + 1, len(nums) - nums.index(max(nums)))
            ),
            max(nums.index(min(nums)) + 1, nums.index(max(nums)) + 1),
            max(len(nums) - nums.index(min(nums)), len(nums) - nums.index(max(nums))),
        )
