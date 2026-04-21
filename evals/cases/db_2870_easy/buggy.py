from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
    s=bin(start)[2:].zfill(50)
        g=bin(goal)[2:].zfill(50)
        count=0
        for i in range(50):
            if s[i]!=g[i]:
                count+=1
    return count
