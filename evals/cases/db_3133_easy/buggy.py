from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def countGoodSubstrings(self, s: str) -> int:
        count = 0
        for i in range(len(s) - 2):
            x = s[i : i + 2]
            if x.count(x[0]) == 1 and x.count(x[1]) == 1 and x.count(x[2]) == 1:
                count += 1
        return count
