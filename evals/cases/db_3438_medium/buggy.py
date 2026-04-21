from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def kthLargestNumber(self, nums: List[str], k: int) -> str:
        ls = []
        for i in nums:
            ls.append(int(i))
        sorted_ = sorted(ls)
        return str(sorted_[k])
