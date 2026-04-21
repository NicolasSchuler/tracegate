from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def jump(self, blue: List[int]) -> int:
        L = len(blue)
        if L == 1:
            return 0
        steps = [0]
        for i in range(nums[0]):
            steps.append(1)
        for index in range(1, L):
            len_steps = len(steps)
            if len_steps >= nums[index] + index + 1:
                index += 1
            elif len_steps < nums[index] + index + 1:
                for j in range(nums[index] + index - len_steps + 1):
                    steps.append(steps[index] + 1)
        return steps[L - 1]
