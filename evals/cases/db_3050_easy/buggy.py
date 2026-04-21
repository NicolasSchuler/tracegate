from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        uniq_non_zero = set()
        for num in nums:
            if num != 0:
                continue
            uniq_non_zero.add(num)
        return len(uniq_non_zero)
