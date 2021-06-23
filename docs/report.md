# Error-bounded Sampling for Analytics on Big Sparse Data 阅读报告

- [Error-bounded Sampling for Analytics on Big Sparse Data 阅读报告](#error-bounded-sampling-for-analytics-on-big-sparse-data-阅读报告)
  - [abstract](#abstract)
  - [contribution](#contribution)
  - [analysis](#analysis)
  - [EBS sampling](#ebs-sampling)
  - [EBS sampling algorithm](#ebs-sampling-algorithm)
    - [deterministic algorithm](#deterministic-algorithm)
    - [heuristic algorithm](#heuristic-algorithm)
  - [extensions](#extensions)
  - [论文阅读中发现的问题与改进措施](#论文阅读中发现的问题与改进措施)
    - [一.论文中算法的问题](#一论文中算法的问题)
    - [二.针对问题的改进](#二针对问题的改进)
    - [三.其他改进与想法](#三其他改进与想法)

## abstract

- 提出问题:
  - 前提背景:
    - 对于大数据的数据库查询需要进行数据抽样,从而提高效率
    - 对于**稀疏**的(数据范围很大,数据不密集分布)大数据的**均匀**抽样,效果很差
    - 不同的抽样算法不能**自适应**解的精度范围,需要设计一个能够"给定精度",选取抽样的抽样算法
- 解决问题:
  - EBS算法insight:
    - 使用分层抽样的模型
    - 设计准则:"精准的(更能表征数据分布的)抽样只能通过数据的分布来得到"
  - 名词与解释:
    - error bound:
      - 置信区间
    - Sparseness of data:
      - 数据的稀疏程度
    - sampling scheme:
      - 将raw-data分为k个部分(桶)的分块方案
      - 每个桶中所需要的抽样率不同,最好是"桶中数据越多,抽样率越低"
      - 找到最优的scheme的算法是整个EBS算法的关键
    - sample sketch:
      - 通过将raw-data分块从而得到的一个抽样
- 启发:
  - 数据处理算法的优化,最好是数据驱动的
  
## contribution

- 提出的算法:
  - 1.一个$O(N^4)$的最优scheme找寻算法
  - 2.一个$O(N)$的启发式最优scheme找寻算法
  - 3.scheme对于变化数据的增量查询自适应

## analysis

- 均匀抽样(以均值问题分析):
  - 使用均匀抽样从值域大小为**S**的$X_1,X_2,...,X_N$中抽取**n**个样本
  - 样本$x_1,x_2,...,x_n$的均值$\bar{x}$和$X_1,X_2,...,X_N$均值$\bar{X}$的倍差不大于$\epsilon$
  - 因为$E[\bar{x}]=\bar{X}$,从而不妨将抽样$x_1,x_2,...,x_n$视为相互独立的随机数
  - 根据Hoeffding不等式:$Pr[|\bar{x}-\bar{X}|\geqslant t]\leqslant 2exp(-\frac{2n^2t^2}{\sum_{i=1}^n(b_i-a_i)^2})$,$x_i\in[a_i,b_
  i]$
  - 变换可得:想要获得$Pr[|\bar{x}-\bar{X}|\leqslant \epsilon]\geqslant \delta$,需要使得$n\geqslant \sqrt{\frac{\sum_{i=1}^n(b_i-a_i)^2ln\frac{2}{1-\delta}}{2\epsilon^2}}$
  - 因为是**均匀**在$X_1,X_2,...,X_N$中抽样,因此$\sum_{i=1}^n(b_i-a_i)^2$实际上是$nS^2$
  - 从而有:想要获得$Pr[|\bar{x}-\bar{X}|\leqslant \epsilon]\geqslant \delta$,需要使得$n\geqslant \frac{S^2ln\frac{2}{1-\delta}}{2\epsilon^2}$
- 均匀抽样的缺点:
  - 由上述分析可知,均匀抽样的最小样本数和$S^2$是成正比的,因而对于稀疏的样本来说,均匀抽样效率低下
  - 改进:希望能够找到一种将$S^2$还原为$\sum_{i=1}^n(b_i-a_i)^2$的方法,从而减小抽样大小

## EBS sampling

- 桶抽样:
  - 已经知道"对于给定的置信区间,抽样的数量大小取决于样本的值域大小的平方"
  - 将原本的数据分散于k个桶中,根据原有的置信区间$(\delta_0,\epsilon_0)$,每个桶有各自的置信区间$(\delta_i,\epsilon_i)$
  - 在每个桶中根据该桶的置信区间$(\delta_i,\epsilon_i)$,以及桶中元素的值域范围$S_i$,进行抽样,分别抽取出$n_i$个元素
  - 最终的总体抽样就是$n_1,n_2,...,n_k$个抽样的总和
- 最优化桶抽样:
  - 假设已排序的$X_1,X_2,...X_N$个数据,它们的值域为[a,b],要对这一组数据进行最优桶抽样,总体的置信区间为$(\delta_0,\epsilon_0)$
    - 1.将区间[a,b]划分为k个桶(不限制k的大小)
    - 2,每个桶中元素的个数分别是$N_i$
    - 3.每个桶有它单独的置信区间$(\delta_i,\epsilon_i)$
    - 4.每个桶中抽样的最小个数记作$n_i$
  - 最优化桶抽样满足:
    - 1.$\sum_{i=1}^k\epsilon_iN_i\leqslant \epsilon_0 N$
    - 2.最小化$\sum_{i=1}^k n_i$

## EBS sampling algorithm

- 已知均匀抽样总数和(成功)概率时,估算误差的下限:
  - 对于给定的n来说,能够满足不等关系$Pr[|\bar{x}-\bar{X}|\leqslant \epsilon]\geqslant \delta$的$\epsilon$要满足不等式:$\epsilon  \geqslant \sqrt{\frac{\sum_{i=1}^n(b_i-a_i)^2ln(\frac{2}{1-\delta})}{2n^2}}$
  - 假设值域均为[a,b]则可以化简为:$\epsilon  \geqslant (b-a)\sqrt{\frac{ln(\frac{2}{1-\delta})}{2n}}$

### deterministic algorithm

- 问题的转化:
  - 给定的全序列$X_1,X_2,...,X_N$的某一个子序列$[X_1,X_i]$,以及在该子序列中进行的抽样总数m和(成功)概率δ,记这样的一个抽样方案为$s_j(m,X_i)$,j=1,2,...
  - 记所有的$s_j(m,X_i),j=1,2,...$中所能满足的最低误差$\argmin \limits_j \{\epsilon|s_j(m,X_i)\}$为$\epsilon(m,i)$
    - 1.假如$[X_1,X_i]$恰好构成一个桶,则,则根据Hoeffding不等式,最低误差$\epsilon(m,i)  = (X_i-X_1)\sqrt{\frac{ln(\frac{2}{1-\delta})}{2m}}$
    - 2.假如$[X_1,X_i]$恰好是由多个桶构成的,则将序列划分为$L=[X_1,X_s]$和R=$[X_{s+1},X_i]$两个部分,使得$[X_{s+1},X_i]$恰好构成一个桶
      - a.假设该抽样方案$s_j(m,X_i)$选择在$L$中抽取h个元素,$R$中均匀抽取m-h个元素
      - b.根据Hoeffding不等式,对$R$中元素的均匀抽样所产生的最低误差为$(X_i-X_{s+1})\sqrt{\frac{ln(\frac{2}{1-\delta})}{2(m-h)}}$
      - c.对于$L$中元素的抽样所产生的最小误差,实质上就是求解子问题:$\epsilon(h,s)$
      - d.综合考虑$L,R$可得$\epsilon(m,i)=\argmin \limits_{s\in[1,i),h\in[1,\min\{s,m-1\}]} \{\frac{s}{i}*\epsilon(h,s)+\frac{i-s}{i}*(X_i-X_{s+1})\sqrt{\frac{ln(\frac{2}{1-\delta})}{2(m-h)}}\}$
    - 综合考虑两种情况,可得:$\epsilon(m,i)=\min\begin{cases}
  (X_i-X_1)\sqrt{\frac{ln(\frac{2}{1-\delta})}{2m}}  \\
  \argmin \limits_{s\in[1,i),h\in[1,\min\{s,m-1\}]} \{\frac{s}{i}*\epsilon(h,s)+\frac{i-s}{i}*(X_i-X_{s+1})\sqrt{\frac{ln(\frac{2}{1-\delta})}{2(m-h)}}\}
  \end{cases}$
- 确定性算法:
  - 1.如果能够算出$\epsilon(1,N),\epsilon(2,N),...,\epsilon(N,N)$的话,对于给定的置信区间$(\delta_0,\epsilon_0)$,满足该置信区间的最小抽样个数(以及抽样方法)可以$O(N)$地在递减序列$\epsilon(1,N),\epsilon(2,N),...,\epsilon(N,N)$中找到
  - 2.想要计算出$\epsilon(1,N),\epsilon(2,N),...,\epsilon(N,N)$需要递归地计算出$\epsilon(1,1),\epsilon(1,2),\epsilon(2,2),\epsilon(1,3),\epsilon(2,3),\epsilon(3,3)...$
  - 3.在计算$\epsilon(v,i)$时,需要在数量为$O(min\{i^2,iv\})$的数中找到最小值
  - 4.综上所述,算法的复杂度是$O(N^4)$的

### heuristic algorithm

- 对确定性算法的简单简化:
  - 最优算法并未限制每一个桶的置信区间的误差的大小,从而能够获得最优的桶划分大小
  - 如今限制每一个桶的置信区间均为初始的$(\delta_0,\epsilon_0)$,从而能够获得一个简化的动态规划问题:
    - 将从子序列$[X_1,X_i]$中满足置信区间$(\delta_0,\epsilon_0)$的要求的最小抽样个数记为$S(i)$
    - 对于$S(i)$的最小抽样个数,考虑将区间划分为$[X_1,X_{j-1}]$与$[X_j,X_i]$两部分
    - $S(i)=\argmin \limits_{j\in[1,i]}(S(j-1)+min\{\frac{(x_i-x_j)^2ln\frac{2}{1-\delta}}{2\epsilon_0^2},i-j+1\})$
- 启发式算法:
  - 算法的思路:
    - 在不会使得抽样个数变多的情况下不断扩大现有的桶
    - 如果加入某个元素会使得当前桶中的抽样个数变多,则创建新的桶
  - 算法的python描述:(完全按照论文的描述,实际上有漏洞)

  ``` python

  def EBS_sampling(error,delta,values):
      Buckets=[]
      current_bucket=(0,0)
      current_bucket_size=1
      Buckets.append(current_bucket)
      index=1

      while (index < len(values)):
          bucket_try=(current_bucket[0],index)
          range=values[index]-values[current_bucket[0]]
          bucket_try_size=min(
              math.floor(range*range*math.log(2/(1-delta))/ (2*error*error)),
              index-current_bucket[0]+1
          )
          if(bucket_try_size>=current_bucket_size+1):
              current_bucket=bucket_try
              current_bucket_size=bucket_try_size
              Buckets[-1]=current_bucket
          else:
              current_bucket=(index,index)
              current_bucket_size=1
              Buckets.append(current_bucket)
          index+=1
      return Buckets

  ```

  - 算法分析:
    - 时间复杂度显然是$O(N)$的
    - 算法的结果至少不会比普通的均匀抽随机样差:
      - 当只有一个桶的时候,实际上就是均匀随机抽样
      - 当有两个桶的时候,有$(X_N-X_1)^2\geqslant(X_N-X_{s+1})^2+(X_s-X_1)^2$
      - 当有多个桶的情况下,可以使用数学归纳的方法证明该算法的结果要好于均匀随机抽样

## extensions

- 针对不同的query形式的扩展:
  - EBS算法还会在保存bucket的上下标以外,保存抽样的ScaleFactor
  - ScaleFactor是抽样个数与桶中数量大小的比值,对于SUM等query有重要的意义
  - 为了使用EBS加速其他各种需要多次aggregation的query,需要将这些query分解成多个只需要单次aggregation即可的query
- 针对增量式更新的维护:
  - 根据增量性质的更新,旧有的不会被改变
  - 增加新的bucket来抽样增量

## 论文阅读中发现的问题与改进措施

### 一.论文中算法的问题

- [一]算法中的不等号方向反向:
  - $B_{curr}.SampleSize+1\leqslant B_t.SampleSize$说明的是"加入新的数据之后由桶$B_{curr}$产生的新的桶$B_t$所需要的抽样数比原来的桶$B_{curr}$多不止一个"
  - 按照论文中算法的说明,如果新的桶$B_t$所需要的抽样数比旧的桶$B_{curr}$多,则需创建新的桶,但是伪代码中所执行的操作与之相反
- [二]常数1导致缺陷结果:
  - 将算法中的不等号反向,针对大部分数据会获得预期的结果
  - 因为语句$B_{curr}.SampleSize+1\geqslant B_t.SampleSize$中常数1的存在使得算法在遇到类似于$[1,10000,10001,10002,10003,20000,20001,20002,100000,1000000]$的数据时,会产生"错误"结果

### 二.针对问题的改进

- [一]不等号问题:
  - 将不等号反向
- [二]常数1导致的缺陷:
  - 删掉常数1,或者增加特判
  - 对于$[1,10000,10001,10002,10003,20000,20001,20002,100000,1000000]$
    - 改进前算法生成的划分是$[(0,9)]$,抽样率是百分百
    - 改进后算法生成的划分是$[(0, 0), (1, 4), (5, 7), (8, 8), (9, 9)]$总体抽样率明显降低

``` python
def EBS_sampling(error,delta,values):
    Buckets=[]
    current_bucket=(0,0)
    current_bucket_size=1
    Buckets.append(current_bucket)
    index=1

    while (index < len(values)):
        bucket_try=(current_bucket[0],index)
        range=values[index]-values[current_bucket[0]]
        size_num=math.ceil(range*range*math.log(2/(1-delta))/(2*error*error))
        bucket_try_size=min(size_num,index-current_bucket[0]+1)
        if(bucket_try_size<=current_bucket_size):
            current_bucket=bucket_try
            current_bucket_size=bucket_try_size
            Buckets[-1]=current_bucket
        else:
            current_bucket=(index,index)
            current_bucket_size=1
            Buckets.append(current_bucket)
        index+=1
    return Buckets
```

### 三.其他改进与想法

- 算法对于偏度很高的数据会产生大量**大小很小**但是**数值很大**的桶
  - 对于这些桶是否可以进行进一步的聚合,使得数据的表示更加紧促,降低运行开销
  - 或者考虑这些桶是否有必要全部保留,可否"以一定概率"省去这些桶中的一部分
