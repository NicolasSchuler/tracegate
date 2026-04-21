from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def flipgame(self, fronts: List[int], backs: List[int]) -> int:
        bad = set(i for i, j in zip(fronts, backs) if i == j)
        for i in sorted(set(fronts - backs)):
            if i in bad:
                continue
            return i
        return 0
