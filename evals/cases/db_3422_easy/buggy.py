from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        l = []
        for i in nums1:
            count = 0
            for j in nums2:
                if i == j:
                    for k in range(nums2.index(j) + 1, len(nums2)):
                        if nums2[k] > j:
                            count += 1
                            l.append(nums2[k])
                            break
                    if count == 0:
                        l.append(j)
        return l
