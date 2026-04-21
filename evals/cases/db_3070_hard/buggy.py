from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools


class Solution:
    def minimumTotalCost(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        z = Counter(nums1)
        z1 = Counter(nums2)
        for i in z:
            if n - z1[i] < z[i]:
                return -1
            if z[i] >= n // 2 + 1 and z1[i] >= n // 2 + 1:
                return -1
        for i in z1:
            if n - z[i] < z1[i]:
                return -1
            if z[i] >= n // 2 + 1 and z1[i] >= n // 2 + 1:
                return -1
        z = Counter([])
        ans = 0
        flag = 0
        d = defaultdict(list)
        vis = [0 for i in range(n)]
        for i in range(n):
            if nums1[i] == nums2[i]:
                z[nums2[i]] += 1
                ans += i
                flag = 1
                d[nums2[i]].append(i)
        t = 0
        l = z.most_common(len(z))
        a = 0
        for i in range(1, len(l)):
            a += l[i][1]
            for j in d[l[i][0]]:
                vis[j] = 1
            z[l[i][0]] = 0
        if l and a >= l[0][1]:
            return ans
        x = 0
        if l:
            x = l[0][1] - a
            z[l[0][0]] = x
        print(z, ans)
        for j in z:
            if z[j]:
                for i in range(n):
                    if vis[i] == 0 and nums1[i] == j and nums2[i] == j and x:
                        if flag:
                            ans += i
                            x -= 1
        return ans
