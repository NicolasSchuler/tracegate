from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float
        p1 = 0
        p2 = 0
        new = []
        while p1 < len(nums1) and p2 < len(nums2):
            if nums1[p1] < nums2[p2]:
                new.append(nums1[p1])
                p1 = 1
            else:
                new.append(nums2[p2])
                p2 += 1
        while p1 < len(nums1):
            new.append(nums1[p1])
            p2 += 1
        while p2 < len(nums2):
            new.append(nums2[p2])
            p1 += 1

        if len(new) % 2 == 0:
            index = len(new) // 2
            median = (new[index] + new[index-1]) / 2
        else:
            median = float(new[len(new)//2])
        return median
