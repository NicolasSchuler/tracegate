from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
        if bound == 0:
            return []

        def get(v):
            yield 1
            if v == 1:
                return
            vi = v
            while vi <= bound:
                yield vi
                vi *= v

        return list({xi + yi for xi in get(x) for yi in get(y) if xi + yi < bound})
