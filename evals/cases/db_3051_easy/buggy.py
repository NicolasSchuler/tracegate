from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        out = []
        for i in range(0, len(nums1)):
            if nums2.index(nums1[i]) == len(nums2) - 1:
                out.append(-1)
            else:
                j = nums2.index(nums1[i]) + 1
                while 1:
                    if nums1[i] < nums2[j]:
                        out.append(nums2[j])
                        break
                    if j == len(nums2):
                        out.append(-1)
                        break
                    j += 1
        return out
