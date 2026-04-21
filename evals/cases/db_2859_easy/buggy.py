from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class NumArray:

    def __init__(self, nums: List[int]):
    self.nums=[0]+list(accumulate(nums))
    print(self.nums)

    def sumRange(self, left: int, right: int) -> int:
        return self.nums[right+1]-self.nums[left]

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)
