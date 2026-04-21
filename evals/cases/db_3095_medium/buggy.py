from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def findLonely(self, nums: List[int]) -> List[int]:
        dic = {}
        res = []
        for i in nums:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1
        for i in nums:
            if dic[i] == 1:
                if (i not in dic) and (i + 1 not in dic):
                    res.append(i)
        return res
