from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def fourSumCount(
        self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]
    ) -> int:
        dictionary = defaultdict(int)
        for n1 in nums1:
            for n2 in nums2:
                numberNeeded = -(n1 + n2)
                dictionary[numberNeeded] += 1

        numberOfTuples = 0
        for n3 in nums3:
            for n4 in nums4:
                numberOfTuples += self.nonExistingMethod(dictionary[n3 + n4])

        return numberOfTuples
