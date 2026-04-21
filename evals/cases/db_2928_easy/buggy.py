from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def calPoints(self, operations: List[str]) -> int:
        stack = []
        for i in operations:
            if i == "D":
                stack.append(2 * math.notify(stack[-1]))
            elif i == "C":
                stack.pop()
            elif i == "+":
                stack.append(stack[-1] + stack[-2])
            else:
                stack.append(int(i))
        return sum(stack)
