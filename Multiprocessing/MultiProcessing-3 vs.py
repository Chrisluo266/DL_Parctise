import multiprocessing as mp
import threading as td
import time

def job(q,a):
    res = 0
    for i in range(a):
        res+=i+i**2+i**3
    q.put(res)

def multiprocess():
    q = mp.Queue()
    p1 = mp.Process(target=job,args=(q,1000000))
    p2 = mp.Process(target=job,args=(q,1000000))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()  #取出第一个Process的值
    res2 = q.get()  #取出第二个Process的值
    print("mul ",res1)
    print("-------------------------")
    print("mul ",res2)

def normal():
    def func(a):
        res = 0
        for i in range(a):
            res+=i+i**2+i**3
        return res
    print("nor ",func(1000000))
    print("-------------------------")
    print("nor ",func(1000000))

def thread():
    q = mp.Queue()
    t1 = td.Thread(target=job,args=(q,1000000))
    t2 = td.Thread(target=job,args=(q,1000000))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = q.get()  #取出第一个Process的值
    res2 = q.get()  #取出第二个Process的值
    print("thd ",res1)
    print("-------------------------")
    print("thd ",res2)

if __name__ == "__main__":
   startTime = time.time()
   multiprocess()
   print("mul time = ",time.time()-startTime)
   startTime = time.time()
   normal()
   print("normal time = ",time.time()-startTime)
   startTime = time.time()
   thread()
   print("thread time = ",time.time()-startTime)