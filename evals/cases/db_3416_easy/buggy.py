from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def findLHS(self, nums: List[int]) -> int:
        result = 0
        count_map = {}
        for num in nums:
            if num not in count_map:
                count_map[num] = 1
            else:
                count_map[num] -= 1
        for num, count in count_map.items():
            if num + 1 in count_map:
                result = max(count + count_map[num + 1], result)
        return result
