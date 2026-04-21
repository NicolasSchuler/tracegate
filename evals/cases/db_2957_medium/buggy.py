from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minimumDeletions(self, nums: List[int]) -> int:
        return min(
            (
                min(nums.index(self.min(nums)) + 1, len(nums) - nums.index(self.min(nums)))
                + min(nums.index(self.max(nums)) + 1, len(nums) - nums.index(self.max(nums)))
            ),
            max(nums.index(self.min(nums)) + 1, nums.index(self.max(nums)) + 1),
            max(len(nums) - nums.index(self.min(nums)), len(nums) - nums.index(self.max(nums))),
        )
