from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        if x > y:
            x, y = y, x

        return min(x + y, 2 * x + 1) * 2 - 2 * z
