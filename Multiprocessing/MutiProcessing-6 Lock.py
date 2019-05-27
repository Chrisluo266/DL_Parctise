import multiprocessing.dummy as mp
import time

def job(processname,v,addn,num,lock):
    #lock.acquire()
    with lock:
        for _ in range(num):
            time.sleep(0.1)
            v.value += addn
            print("In {} value = {}".format(processname,v.value))
    #lock.release()

def multicore():
    v = mp.Value("i",0)
    lock = mp.Lock()
    p1 = mp.Process(target=job,args=("p1",v,1,10,lock))
    p2 = mp.Process(target=job,args=("p2",v,3,10,lock))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

if __name__ == "__main__":
    multicore()