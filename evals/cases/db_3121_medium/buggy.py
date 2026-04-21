from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
from collections import defaultdict


class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        x = self.recursion(s, k)
        if k == 0:
            return x
        else:
            return 0

    def recursion(self, s, k):
        if len(s) <= 1:
            return len(s)
        flag = True
        for i in range(len(s)):
            if s.count(s[i]) < k:
                flag = False
                break

        if flag:
            return len(s)
        left = self.recursion(s[:i], k)
        right = self.recursion(s[i + 1 :], k)
        if left > right:
            return left
        return right
