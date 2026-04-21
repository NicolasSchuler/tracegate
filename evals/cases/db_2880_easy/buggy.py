from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        count = [0] * 101

    res = []
    for num in nums:
        count[num] += 1
    for num in nums:
        res.append(sum(count[:num]))
    return res
