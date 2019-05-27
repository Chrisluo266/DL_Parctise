import multiprocessing as mp

def job(q,a):
    res = 0
    for i in range(a):
        res+=i+i**2+i**3
    q.put(res)

if __name__ == "__main__":
    q = mp.Queue()
    p1 = mp.Process(target=job,args=(q,100))
    p2 = mp.Process(target=job,args=(q,1000))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()  #取出第一个Process的值
    res2 = q.get()  #取出第二个Process的值
    print(res1)
    print("-------------------------")
    print(res2)