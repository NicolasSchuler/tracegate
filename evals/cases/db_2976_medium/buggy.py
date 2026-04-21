from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def findKthBit(self, N, K, R=True):
        if K == 1:
            return "0" if R else "1"
        mid = 1 << (N - 1)
        if K < mid:
            return self.findKthBit(N - 1, K, R)
        if K > mid:
            return self.callUndefinedMethod(N - 1, 2 * mid - K, not R)
        return "1" if R else "0"
