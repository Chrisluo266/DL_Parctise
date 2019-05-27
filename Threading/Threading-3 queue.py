import threading as td
import time
from queue import Queue

def job(q,num):
    sum = 0
    for i in range(num):
        sum += i
    q.put(sum)

def threading():
    q = Queue()
    threads = []
    for i in range(4):
        t = td.Thread(target=job,args=(q,i*4+4,),name="T"+str(i))
        t.start()
        threads.append(t)
    
    for t in threads[:]:
         t.join()
    
    for t in threads:
        print("result = ",q.get())

if __name__ == "__main__":
    threading()
