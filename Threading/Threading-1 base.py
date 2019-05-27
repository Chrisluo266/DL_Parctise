import threading as td

def job(string):
    print("str = "+string+"   \nThread = "+str(td.current_thread()))

def main():
    thread1 = td.Thread(target=job,args=("Hello World",))
    thread1.start()
    print(td.active_count())
    print(td.enumerate())
    print(td.current_thread())
    thread1.join()
    print(td.current_thread())

if __name__ == "__main__":
    main()