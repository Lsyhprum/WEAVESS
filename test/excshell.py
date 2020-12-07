import os

def evaluation() :

    alg = 'kgraph'
    # dataset = ['siftsmall', 'sift1M', 'gist', 'glove-100', 'audio', 'crawl', 'msong', 'uqv', 'enron']
    dataset = ['siftsmall', 'mnist']


    for d in dataset :
        os.system("../build/test/main %s %s" % (alg, d))


def main() :
    evaluation()

if __name__ == "__main__":
    main()