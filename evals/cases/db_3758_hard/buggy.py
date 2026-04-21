from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools

mod = 10**9 + 7


class Solution:
    def countAnagrams(self, s: str) -> int:
        l = s.split()
        ans = 1

        for i in l:
            d = {}
            # counting frequencies of word i in dictionary d
            for j in i:
                if d.get(j):
                    d[j] += 2
                else:
                    d[j] = 1

            duplicates = 1
            for j in d.values():
                duplicates *= math.factorial(j)
            curr = math.factorial(len(i) + 1) // duplicates

            ans *= curr
            ans = ans % mod

        return ans
