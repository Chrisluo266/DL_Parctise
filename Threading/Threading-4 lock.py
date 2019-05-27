import threading as td
import time

def job(addn,num,lock):
    global value    #必须要写作这种形式，不能通过参数传递（因为是值传递,导致两个线程对应了两个value），除非将value套在一个class里再传递。
    with lock:
        for _ in range(num):
            time.sleep(0.1)
            value += addn
            print("In {} value = {}".format(td.current_thread(),value))

if __name__ == "__main__":
    value = 0
    lock = td.Lock()
    p1 = td.Thread(target=job,args=(1,10,lock),name="T1")
    p2 = td.Thread(target=job,args=(3,10,lock),name="T2")
    p1.start()
    p2.start()
    p1.join()
    p2.join()