from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        s= ''.join(map(str,digits))
        i='int(s)+1
        if i<10:
            li=[]
        else:
            li=list(map(int,str(i)))  
        return li
