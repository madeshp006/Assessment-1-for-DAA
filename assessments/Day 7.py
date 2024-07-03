# 1. Height of Binary Tree After Subtree Removal Queries
class T:
    def __init__(self, x):
        self.v = x
        self.l = None
        self.r = None
def h(r):
    if not r:
        return -1
    return 1 + max(h(r.l), h(r.r))
def g_h(r, q):
    if not r:
        return -1
    if r.v == q:
        return -1
    l_h = g_h(r.l, q)
    r_h = g_h(r.r, q)
    return 1 + max(l_h, r_h)
def g(r, q):
    a = []
    for i in q:
        a.append(h(r) if r.v == i else g_h(r, i))
    return a
root = T(1)
root.l = T(2)
root.r = T(3)
root.l.l = T(4)
root.l.r = T(5)
queries = [1, 2, 3]
print(g(root, queries))

# 2. Sort Array by Moving Items to Empty Space
def m(a):
    c = 0
    p = {v: i for i, v in enumerate(a)}
    t = list(range(len(a)))
    for i in range(len(a)):
        if a[i] != t[i]:
            j = p[t[i]]
            a[i], a[j] = a[j], a[i]
            p[a[j]] = j
            c += 1
    return c
array = [4, 3, 2, 1, 0]
print(m(array))

# 3. Apply Operations to an Array
def a(a):
    n = len(a)
    for i in range(n-1):
        if a[i] == a[i+1]:
            a[i] *= 2
            a[i+1] = 0
    a.sort(key=lambda x: x == 0)
    return a
arr = [2, 2, 3, 3, 3, 4]
print(a(arr))

# 4. Maximum Sum of Distinct Subarrays With Length K
def s(a, k):
    m = 0
    for i in range(len(a) - k + 1):
        s_a = a[i:i+k]
        if len(s_a) == len(set(s_a)):
            m = max(m, sum(s_a))
    return m
array = [1, 2, 3, 4, 5, 6, 1, 2]
k = 3
print(s(array, k))

# 5. Total Cost to Hire K Workers
def w(a, k, c):
    import heapq
    t = 0
    l = a[:c]
    r = a[-c:]
    heapq.heapify(l)
    heapq.heapify(r)
    for _ in range(k):
        if l and r:
            if l[0] <= r[0]:
                t += heapq.heappop(l)
            else:
                t += heapq.heappop(r)
        elif l:
            t += heapq.heappop(l)
        else:
            t += heapq.heappop(r)
    return t
costs = [1, 3, 5, 7, 9]
k = 3
c = 2
print(w(costs, k, c))

# 6. Minimum Total Distance Traveled
def d(r, f):
    r.sort()
    f.sort()
    dp = [[float('inf')] * (len(f) + 1) for _ in range(len(r) + 1)]
    dp[0][0] = 0
    for i in range(1, len(r) + 1):
        for j in range(1, len(f) + 1):
            dp[i][j] = dp[i][j-1]
            if j >= i:
                dp[i][j] = min(dp[i][j], dp[i-1][j-1] + abs(r[i-1] - f[j-1][0]))
    return min(dp[-1])
robots = [1, 3, 5]
factories = [[2, 1], [4, 2]]
print(d(robots, factories))

# 7. Minimum Subarrays in a Valid Split
def s_v(a):
    from math import gcd
    n = len(a)
    dp = [float('inf')] * n
    dp[0] = 1
    for i in range(1, n):
        for j in range(i):
            if gcd(a[j], a[i]) > 1:
                dp[i] = min(dp[i], dp[j] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1
array = [2, 3, 4, 9, 8]
print(s_v(array))

# 8. Number of Distinct Averages
def a(a):
    s = set()
    while a:
        m = min(a)
        x = max(a)
        a.remove(m)
        a.remove(x)
        s.add((m + x) / 2)
    return len(s)
array = [1, 2, 3, 4]
print(a(array))

# 9. Count Ways To Build Good Strings
def c_g_s(z, o, l, h):
    m = 10**9 + 7
    dp = [0] * (h + 1)
    dp[0] = 1
    for i in range(1, h + 1):
        if i >= z:
            dp[i] = (dp[i] + dp[i - z]) % m
        if i >= o:
            dp[i] = (dp[i] + dp[i - o]) % m
    
    return sum(dp[l:h + 1]) % m
low = 3
high = 3
zero = 1
one = 1
print(c_g_s(zero, one, low, high))

# 10. Most Profitable Path in a Tree
from collections import defaultdict, deque
def m_p_p(edges, bob, amount):
    n = len(amount)
    tree = defaultdict(list)
    for u, v in edges:
        tree[u].append(v)
        tree[v].append(u)
    def bfs(start):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            for neighbor in tree[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    q.append(neighbor)
        return dist
    dist_from_root = bfs(0)
    dist_from_bob = bfs(bob)
    max_income = float('-inf')
    def dfs(node, parent, income):
        nonlocal max_income
        is_leaf = True
        for neighbor in tree[node]:
            if neighbor != parent:
                is_leaf = False
                if dist_from_root[neighbor] > dist_from_bob[neighbor]:
                    dfs(neighbor, node, income + amount[neighbor])
                elif dist_from_root[neighbor] == dist_from_bob[neighbor]:
                    dfs(neighbor, node, income + amount[neighbor] // 2)
                else:
                    dfs(neighbor, node, income)
        if is_leaf:
            max_income = max(max_income, income)
    if dist_from_root[0] > dist_from_bob[0]:
        dfs(0, -1, amount[0])
    elif dist_from_root[0] == dist_from_bob[0]:
        dfs(0, -1, amount[0] // 2)
    else:
        dfs(0, -1, 0)
    return max_income
edges = [[0, 1], [1, 2], [1, 3], [3, 4]]
bob = 3
amount = [-2, 4, 2, -4, 6]
print(m_p_p(edges, bob, amount))
