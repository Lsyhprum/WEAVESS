import os
import time

def evaluation() :

    alg = 'fanng'
    dataset = ['audio', 'crawl']
    # dataset = ['siftsmall', 'mnist']

    for d in dataset :
        os.system("../build/test/main %s %s" % (alg, d))

def get_all_pid() :
    return [d for d in os.listdir("/proc") if d.isdigit()]

def isinstance(pids, pid) :
    for p in pids :
        if (p == pid) :
            return 1
    return 0

if __name__ == "__main__" :
    pid = '38534'
    while isinstance(get_all_pid(), pid) :
        # current_time = time.asctime(time.localtime(time.time()))
        # print('%s is running..., current time: %s' % (pid, current_time))
        time.sleep(60)
    
    # print('%s is over' % pid)
    evaluation()