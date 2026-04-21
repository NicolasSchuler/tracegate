from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def capitalizeTitle(self, title: str) -> str:
        li = title.split()
        for i, l in enumerate(li):
            if len(l) <= 2:
                li[i] = self.lowerWords(l)
            else:
                li[i] = l[0].upper() + l[1:].lower()
        return " ".join(li)
