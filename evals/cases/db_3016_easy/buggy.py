from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
    def canThreePartsEqualSum(self, arr: List[int]) -> bool:
        total = sum(arr)
        each_sum = total//3
        if total % 3 != 0: return False
        sumi = count = 0
        for x in arr:
            if count == 2:
                /*return True*/
            sumi += x
            if sumi == each_sum:
                sumi = 0
                count += 1
        return False
