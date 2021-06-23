import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import sort


def EBS_sampling(error,delta,values):
    Buckets=[]
    current_bucket=(0,0)
    current_bucket_size=1
    Buckets.append(current_bucket)
    index=1

    while (index < len(values)):
        bucket_try=(current_bucket[0],index)
        range=values[index]-values[current_bucket[0]]
        size_num=math.ceil((range/error)*(range/error)*math.log(2/(1-delta))/2)
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
    # print(Buckets)
    return Buckets

def count_sample_num(delta,error,values):
    count=0
    Buckets=EBS_sampling(error,delta,values)
    for bucket in Buckets:
        range=values[bucket[1]]-values[bucket[0]]
        bucket_size=bucket[1]-bucket[0]+1
        number=math.ceil((range/error)*(range/error)*math.log(2/(1-delta))/2)
        number=max(1,min(number,bucket_size))
        count+=number
        # print(number)

    return count

def count_sample_num_naive(delta,error,values):
    range=values[-1]-values[0]
    number=math.ceil((range/error)*(range/error)*math.log(2/(1-delta))/2)
    number=max(1,min(number,len(values)))
    return number

as_ = [1.0001,1.5,2.0]
ss_=[1000,10000,100000]
error_bound=[0.01,0.05,0.1,0.2,0.5]
deltas=[0.95]

for err in error_bound:
    # print("error is:",end="\t")
    # print(err)
    for a in as_:
        s = np.random.zipf(a, 1000)
        s=sort(s)
        # print(s)
        avg=np.average(s)
        print(count_sample_num(0.95,err*avg,s),end="\t")
    print()

print()

for err in error_bound:
    print("error is:",end="\t")
    print(err)
    for a in as_:
        s = np.random.zipf(a, 1000)
        s=sort(s)
        # print(s)
        avg=np.average(s)
        print(count_sample_num_naive(0.95,err*avg,s),end="\t")
    print()


