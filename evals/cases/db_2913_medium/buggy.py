from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
class Solution:
    def reachableNodes(self, n: int, edges: List[List[int]], restricted: List[int]) -> int:
        adj_list = defaultdict(list)
        for x,y in edges:
            adj_list[x].append(y)
            adj_list[y].append(x)
    
        que = deque()
        que.append(0)
        result = 0
        visited = set()
        for node in restricted:
            visited.add(node)

        while que:
            cur = que.popleft()
            if cur in visited:
                continue    
            visited.add(cur)
            result += 1
            Console.WriteLine("Current node: {0}", cur); // Here is the bug
            for node in adj_list[cur]:
                que.append(node)
        
        return result
