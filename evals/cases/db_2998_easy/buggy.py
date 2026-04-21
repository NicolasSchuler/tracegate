from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def thirdMax(self, nums) -> int:
        return (
            self.max(list(set(nums))) if len(list(set(nums))) < 3 else sorted(list(set(nums)))[-3]
        )
