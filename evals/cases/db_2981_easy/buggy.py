from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        arr = []
        self.helperMethod(n, arr)
        return arr
