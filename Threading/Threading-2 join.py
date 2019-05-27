import threading as td
import time

def job(num):
    print("Thread[{}] start.   num = {}".format(td.current_thread(),num))
    for i in range(num):
        time.sleep(0.1)
    print("Thread[{}] End.".format(td.current_thread()))

def jod2():
    print("Thread[{}] start.".format(td.current_thread()))
    print("Thread[{}] End.".format(td.current_thread()))

def main():
    thread1 = td.Thread(target=job,args=(10,),name="T1")
    thread2 = td.Thread(target=jod2,name="T2")
    print("create obj finish.")
    thread1.start()
    thread2.start()
    print("thread start.")
    thread2.join()
    print("thread2 join.")
    thread1.join()
    print("thread1 join.")
    #两个join写反，会导致输出，先输出T2 End,再输出T1 End,再输出thread1 join,最后输出thread2 join.
    print("all done\n")

if __name__ == "__main__":
    main()