import os

def evaluation() :

    num_threads = '2'
    dataset_root = '/home/yq/ANNS/dataset/'
    # dataset = ['sift1M', 'gist', 'glove-100', 'audio', 'crawl', 'msong', 'uqv', 'enron']
    dataset = ['siftsmall']
    base_tail = '_base.fvecs'
    query_tail = '_query.fvecs'
    ground_tail = '_groundtruth.ivecs'

    for d in dataset :
        if d == 'sift1M' :
            base_path = dataset_root + d + '/sift' + base_tail
            query_path = dataset_root + d + '/sift' + query_tail
            ground_path = dataset_root+ d + '/sift' + ground_tail
        else :
            base_path = dataset_root + d + '/' + d + base_tail
            query_path = dataset_root + d + '/' + d + query_tail
            ground_path = dataset_root + d + '/' + d + ground_tail
        LSHtable_path = 'LSHtable_' + d + '.txt'
        LSHfunc_path = 'LSHfunc_' + d + '.txt'
        # print(base_path, query_path, ground_path, LSHtable_path, LSHfunc_path, num_threads)
        os.system("matlab -nodisplay -r \'LSHrun %s %s %s %s %s %s %s;quit;\'" % (d, base_path, query_path, ground_path, LSHtable_path, LSHfunc_path, num_threads))


def main() :
    evaluation()

if __name__ == "__main__":
    main()