from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minimumPartition(self, s: str, k: int) -> int:
        curr, ans = 0, 1
        for d in s:
            if int(d) > k:
                return -1
            curr = 10 * curr + int(d)
            if curr > k:
                ans += 1
                curr = int(d)
        return curr
