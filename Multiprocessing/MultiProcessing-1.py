import multiprocessing as mp
#import threading as td

def job(str):
    print("In Job. print %s"%(str))

#task1 = td.Thread(target=job,args=("Hello Thread."))
if __name__ == "__main__":
    process1 = mp.Process(target=job,args=("Hello Process.",))

    process1.start()
    process1.join()