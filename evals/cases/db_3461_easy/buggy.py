from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        res = []
        for i in range(left, right + 1):
            ch = True
            for j in str(i):
                if j != "0":
                    if i % int(j) != 0:
                        ch = False
                else:
                    ch = False
                    break
            if ch:
                res.append(i)
        return res[-1]
