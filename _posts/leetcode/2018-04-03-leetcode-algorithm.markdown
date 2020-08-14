---
layout:     post
title:      "滑动窗口题型练习"
subtitle:   ""
date:       2018-04-03 12:00:00
author:     "zhihao"
header-img: "img/post-bg-js-module.jpg"
tags:
    - leetcode
    - python
---

## 滑动窗口

场景：如果我们找到了一个满足要求的区间，并且当区间的右边界再向右扩张已没有意义，此时可以移动左边界到达不满足要求的位置。再移动右边界，持续如此，直到区间的右边界到达整体的结束点

### 1. Longest Substring Without Repeating Characters
> Given a string, find the length of the longest substring without repeating characters.

```
Example 1:
Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 

Example 2:
Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 

Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
```
运用队列，在新增遇到重复字符前，记录并更新最长长度，遇到时队列另一测就开始出队，直到遇到重复字符为止。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # solution: 队列 (滑动窗口)
        result = []
        max = 0
        for element in s:
            if len(result) == 0 or element not in result:
                result.append(element)
                cur = len(result)
                if cur > max:
                    max = cur
            else:
                while result[0] != element:

                    result.remove(result[0])

                result.remove(element)
                result.append(element)

        return max
```

### 2. Minimum Window Substring
> Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

```
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"
```
**note:**
* If there is no such window in S that covers all characters in T, return the empty string "".
* If there is such window, you are guaranteed that there will always be only one unique minimum window in S.

双指针思想，尾指针不断往后扫，当扫到有一个窗口包含了所有T的字符，然后再收缩头指针，直到不能再收缩为止。最后记录所有可能的情况中窗口最小的。

```python
from collections import Counter


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # better solution: improved sliding window。通过只用字符串s中出现在字符串t中的字符（注意记录下标）来创建window，来
        # 减少循环比较次数。
        # 巧妙之处在于，虽然需过滤没有出现在t中的字符，但这并不意味着在s上直接删除字符，而是用一个字典记录所有出现在s和t中的字符，并记录
        # 这些字符在s里面的相应位置。

        # special consideration: s or t is empty
        if not t or not s:
            return ""

        # create a new "S"
        filtered_S = []
        for i, char in enumerate(s):
            if char in t:
                filtered_S.append((i, char))

        # create two pointers and other parts
        left, right = 0, 0
        formed = 0
        window = {}
        dict_t = Counter(t)
        ans = (float("inf"), None, None)

        # move right pointers
        while right < len(filtered_S):
            # add elements
            window[filtered_S[right][1]] = window.get(filtered_S[right][1], 0) + 1

            # decide whether the current window satisfies the requirement
            if window.get(filtered_S[right][1]) == dict_t.get(filtered_S[right][1]):
                formed += 1

            # decide whether to move the left pointer
            while formed == len(dict_t):
                # compare the length and record it
                if ans[0] > filtered_S[right][0] - filtered_S[left][0] + 1:
                    ans = (filtered_S[right][0] - filtered_S[left][0] + 1, filtered_S[left][0], filtered_S[right][0])

                # reduce the number of the specified element
                window[filtered_S[left][1]] -= 1

                # decide whether to reduce the form
                if window.get(filtered_S[left][1]) < dict_t.get(filtered_S[left][1]):
                    formed -= 1

                left += 1

            right += 1

        return "" if ans[0] == float("inf") else s[ans[1]: ans[2]+1]
```






## 动态规划

#### 最长公共子串

```python

def find_lcsubstr(s1, s2): 
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0   #最长匹配的长度
    p=0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p],mmax   #返回最长子串及其长度
 
print find_lcsubstr('abcdfg','abdfg')
```

0-1背包

```python
capacity=10 #背包能容纳的weight
n=5  #商品种类
weights=[2,1,3,3,4]
val=[4,2,6,7,6]
def knapsack(capacity,n,weights,val):
    dp=[0]*(capacity+1)
    for i in range(1,n+1):
        w,v=weights[i-1],val[i-1]
        for j in range(capacity,0,-1):
            if(j>=w):
                dp[j]=max(dp[j],dp[j-w]+v)
    return dp[capacity]

print(knapsack(capacity,n,weights,val))
```

完全背包（物品数量为无限个）
```python
capacity=10 #背包能容纳的weight
n=5  #商品种类
weights=[2,1,3,3,4]
val=[4,2,6,7,6]
def knapsack(capacity,n,weights,val):
    dp=[0]*(capacity+1)
     for i in range(0,n):
        w,v=weights[i],val[i]
        for j in range(w,capacity+1):
            dp[j]=max(dp[j],dp[j-w]+v)
    return dp[capacity]

print(knapsack(capacity,n,weights,val))
```

322.coins change
```python
def DP_Com(exlist,num):
  an=[1]+[0]*num
  for i in exlist :
      for j in range(i,num+1):
          an[j]+=an[j-i]
  return an[num]
print(DP_Com([1,2,5,10,20,50],100))
```
最少需要几张coin
```python
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # dp[i]表示amount=i需要的最少coin数
        dp = [0]+[amount+1] * amount
        for i in range(amount+1):
            for c in coins:
                    if c <= i:
                        dp[i] = min(dp[i], dp[i-c]+1)
        return dp[amount] if dp[amount] <= amount else -1
```