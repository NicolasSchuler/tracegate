from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def totalMoney(self, n: int) -> int:
        income = 1
        a = [1]
        ans = 0
        for i in range(n):
            if i % 7 == 0 and i != 0:
                income = a[len(a) - 8]
            ans += income
            income += 1
            a.append(income)
        return ans
