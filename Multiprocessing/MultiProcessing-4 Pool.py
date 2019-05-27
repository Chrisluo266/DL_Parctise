import multiprocessing as mp

def job(a):
    return a**3
def pow(a,b):
    return a**b

def multiprocess():
    #用pool的话就可以用return，不用pool就只能讲结果放入Queue中
    pool = mp.Pool()
    res = pool.map(job,range(1,11,1))  #map的参数2是一个可迭代对象，且只能支持单参数
    print("1-10 **3 res = ",res)

    res = pool.map(job,(5,6))
    print("5**3|6**3 res = ",res)

    res = pool.apply_async(pow,(5,6))  #apply_async的参数2是整个参数列表
    print("5**6 res = ",res.get())

    multi_res = [pool.apply_async(job,(i,)).get() for i in range(1,11,1)]
    print(multi_res)
    multi_res = [pool.apply_async(pow,(i,3)).get() for i in range(1,11,1)]
    print(multi_res)


if __name__ == "__main__":
   multiprocess()