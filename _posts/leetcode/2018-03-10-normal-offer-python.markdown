---
layout:     post
title:      "剑指offer-python解题"
subtitle:   ""
date:       2018-03-10 12:00:00
author:     "zhihao"
header-img: "img/post-bg-js-module.jpg"
tags:
    - leetcode
    - python
---


##### 二分查找
 ```
def  bin_search(data, val):
    l= 0
    h= len(list) - 1
    while l<= h:
        mid =l+ (h - l) / 2
        if data[mid] == val:
            return mid
        elif data[mid] > val：
            h= mid - 1
        else:
            l= mid + 1
    return -1
```
##### 快排
```
def quicksort(array):
    if len(array)<2:
        return array
    else:
        pivot=array[0]
        less=[i for i in array[1:] if i<=pivot]
        greater=[i for i in array[1:] if i>pivot]
        return quicksort(less)+[pivot]+quicksort(greater)

print(quicksort([5,3,24,6,7,1,3,9,2]))

```
##### 二叉树的镜像
```
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root==None:
            return
        root.left,root.right=root.right,root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```
##### 链表中环的入口结点 
```
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow=fast=head
        while fast and fast.next:
            slow=slow.next
            fast=fast.next.next
            if slow==fast:
                fast=head
                while slow!=fast:
                    slow=slow.next
                    fast=fast.next
                return fast
        return None
```
##### 两个栈实现队列
```
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        self.stack1.append(node)
    def pop(self):
        if not self.stack1:
            return None
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        res = self.stack2.pop()
        while self.stack2:
            self.stack1.append(self.stack2.pop())
        return res

if __name__ == '__main__':
    s=Solution()
    list=[1,2,3,4,5]
    for i in range(len(list)):
        s.push(list[i])
    for i in range(len(list)):
        print(s.pop())
```
##### 反转单链表
```
class Solution:
    """
    @param head: The first node of the linked list.
    @return: You should return the head of the reversed linked list.
                  Reverse it in-place.
    """
    def reverse(self, head):
        # write your code here
        if head is None: return None
        p = head
        cur = None
        pre = None
        while p is not None:
            cur = p.next
            p.next = pre
            pre = p
            p = cur
        return pre
```
##### 和为S的两个数字且乘积最小
```
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        dic=dict()
        ret=[]
        for num in array:
            if tsum-num in dic:
                if ret==[]:
                    ret=[tsum-num,num]
                elif ret[0]*ret[1]>(tsum-num)*num:
                    ret=[tsum-num,num]
            else:
                dic[num]=1
        return ret
```
##### 顺时针打印矩阵
```
class Solution:
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        result = []
        while (matrix):
            result += matrix.pop(0)
            if not matrix or not matrix[0]:
                break
            matrix = self.turn(matrix)
        return result

    def turn(self, matrix):
        row = len(matrix)
        col = len(matrix[0])
        newMatrix = []
        for i in range(col):
            sb = []
            for j in range(row):
                sb.append(matrix[j][i])
            newMatrix.append(sb)
        newMatrix.reverse()
        return newMatrix

if __name__ == '__main__':
    arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    print(Solution().spiralOrder(arr))
```
##### 之字形打印二叉树（如果是顺序打印就删掉flag相关逻辑）
```
class Solution:
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        nodeStack=[pRoot]
        result=[]
        flag=True
        while nodeStack:
            flag= not flag
            nextStack=[]
            tmp=[]
            for i in nodeStack:
                tmp.append(i.val)
                if i.left:
                    nextStack.append(i.left)
                if i.right:
                    nextStack.append(i.right)
            nodeStack=nextStack
            if flag:
                tmp.reverse()
            result.append(tmp)
        return result
```
##### 先序中序重建二叉树
```
class Solution(object):
    def buildTree(self, preorder, inorder):
        if len(inorder)>0:    
            mid=inorder.index(preorder.pop(0))
            root=TreeNode(inorder[mid])
            root.left=self.buildTree(preorder,inorder[:mid])
            root.right=self.buildTree(preorder,inorder[mid+1:])
            return root
```
##### 中序后序重建二叉树
```
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if len(inorder)>0:
            mid=inorder.index(postorder.pop(len(postorder)-1))
            root=TreeNode(inorder[mid])
            root.right=self.buildTree(inorder[mid+1:],postorder)
            root.left=self.buildTree(inorder[:mid],postorder)
            return root
```
##### 滑动窗口的最大值
给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。 
```
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if size == 0:
            return []
        ret = []
        stack = []
        for pos in range(len(num)):
            while (stack and stack[-1][0] < num[pos]):
                stack.pop()
            stack.append((num[pos], pos))
            if pos >= size - 1:
                while (stack and stack[0][1] <= pos - size):
                    stack.pop(0)
                ret.append(stack[0][0])
        return ret


if __name__ == '__main__':
    num = [2, 3, 4, 2, 6, 2, 5, 1]
    size = 3
    print(Solution().maxInWindows(num, size))
```
##### 判断一个数是不是丑数
```
class Solution(object):
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num <= 0:
            return False
        for i in [2,3,5]:
            while num % i == 0:
                num = num / i
        if num == 1:
            return True
        else:
            return False

if __name__ == '__main__':
    print(Solution().isUgly(14))
```
##### [两个链表的第一个公共结点](https://www.nowcoder.com/practice/6ab1d9a29e88450685099d45c9e31e46?tpId=13&tqId=11189&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking) 
思考：设链表pHead1的长度为a,到公共结点的长度为l1；链表pHead2的长度为b,到公共结点的长度为l2，有a+l2 = b+l1
```
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if pHead1== None or pHead2 == None:
            return None
        pa = pHead1
        pb = pHead2 
        while(pa!=pb):
            pa = pHead2 if pa is None else pa.next
            pb = pHead1 if pb is None else pb.next
        return pa
```
##### [最小的K个数](https://www.nowcoder.com/practice/6a296eb82cf844ca8539b57c23e6e9bf?tpId=13&tqId=11182&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

```
import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        heaps = []
        ret = []
        for num in tinput:
            heapq.heappush(heaps,num)
        if k>len(heaps):
            return []
        for i in range(k):
            ret.append(heapq.heappop(heaps))
        return ret
```
##### 二维数组中的查找
```
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        if not array:
            return False
        i = 0
        j = len(array[0])-1
        while(i<len(array) and j>=0):
            if array[i][j]==target:
                return True
            elif array[i][j]>target:
                j-=1
            else:
                i+=1
        return False
```
##### 替换空格
```
import re
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        return re.sub(" ","%20",s)
```
##### 是否为平衡二叉树
```
class Solution(object):
    def isBalanced(self, root):
        def height(node):
            if not node:return 0
            left = height(node.left)
            right = height(node.right)
            if left == -1 or right == -1 or abs(left-right) > 1:
                return -1

            return max(left,right) + 1
        return height(root) != -1
```
##### 字符串全排列
```
class Solution:
    def Permutation(self, ss):
        if not ss:
            return []
        res = []
        self.helper(ss, res, '')
        return sorted(list(set(res)))

    def helper(self, ss, res, path):
         if not ss:
            res.append(path)
         else:
            for i in range(len(ss)):
                self.helper(ss[:i] + ss[i+1:], res, path + ss[i])

if __name__ == '__main__':
    print(Solution().Permutation("abcd"))
```
##### 合并两个排序的链表
```
class Solution:
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 and not l2:
            return None
        dummy = ListNode(0)
        cur = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next
```
##### 二叉树中和为某一值的路径(dfs)
```
class Solution:
    def __init__(self):
        self.li = []
        self.liall = []

    def FindPath(self, root, expectNumber):
        if root is None:
            return self.liall
        self.li.append(root.val)
        expectNumber -= root.val
        if expectNumber == 0 and not root.left and not root.right:
            self.liall.append(self.li[:])
        self.FindPath(root.left, expectNumber)
        self.FindPath(root.right, expectNumber)
        self.li.pop()
        return self.liall
```
##### 二进制中1的个数
```
class Solution:
    def numberof1(self,n):
        count=0
        while n:
            n=n&(n-1)
            count+=1
        return count

```
##### [跳台阶](https://www.nowcoder.com/practice/8c82a5b80378478f9484d87d1c5f12a4?tpId=13&tqId=11161&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking) 
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。 

```
class Solution:
    def jumpFloor(self, number):
        # write code here
        '''
        n = 1 : 1 
        n = 2 : 1+1 = 2
        n = 3 : dp[n-2]+dp[n-1]
        '''
        if number == 1 or number == 2:
            return number
        dp = [1,2]
        for i in range(number-2):
            dp.append(dp[-1]+dp[-2])
        return dp[-1]
```
##### [变态跳台阶](https://www.nowcoder.com/practice/22243d016f6b47f2a6928b4313c85387?tpId=13&tqId=11162&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking) 
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法
```
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number == 1 or number == 2:
            return number
        ret = sum_ = 3
        for i in range(number-2):
            ret = sum_+1
            sum_+=ret
        return ret 
```

##### 连续子数组最大和
```
class Solution:
        def maxSubArra(self,nums):
            cursum=maxsum=nums[0]
            for i in range(1,len(nums)):
                cursum=max(nums[i],cursum+nums[i])
                maxsum=max(cursum,maxsum)
            return maxsum
```