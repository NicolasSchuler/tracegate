from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def findKthPositive(self, arr: [], k: int) -> int:
        l, h = 0, len(arr)
        while l < h:
            mid = (h + l) // 2
            if arr[mid] - mid > k:
                h = mid
            else:
                l = mid + 1
        return self.undefined_method(l + k)
