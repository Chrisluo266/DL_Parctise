import multiprocessing.dummy as mp

value = mp.Value('i',1)  #第一个参数是类型，第二个参数是值
array = mp.Array('i',[1,2,3,4,5])


