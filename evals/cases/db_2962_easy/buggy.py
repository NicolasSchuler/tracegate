from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        l = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                l.append("FizzBuzz")

            elif i % 3 != 0 and i % 5 != 0:
                l.append(str(i))
            elif i % 3 == 0:
                l.append(self.fizz())
            elif i % 5 == 0:
                l.append(self.buzz())

        return l
