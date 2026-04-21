from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
    def makeSmallestPalindrome(self, s: str) -> str
        before = 0
        after = len(s)-1
        l = [i for i in s]
        while before <= len(s):
            l[before] = self.min(l[before], l[after])
            l[after] = l[before]
            before+=1
            after+=1
        return "".join(l)
