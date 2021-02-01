import re
import numpy
import h5py
import struct
 

def load_txt_file(txt_filename):
    data = []

    print("begin load data")
    with open(txt_filename, 'r') as f:
        line_num = 0
        for line in f:
            line_num += 1
            if line_num <= 1000000:
                vector = []
                for word in re.split(' ', line):
                    vector.append(float(word))
                data.append(vector)
    print("load data finish")
    return data

def load_hdf5_file(filename):
    f = h5py.File(filename, 'r')
    print(list(f.keys()))
    #print('dataset')
    #print('dim : {0}'.format(len(f['dataset'][0])))
    # print(f['dataset'][0:10])
    # print(type(f['dataset'][0][0]))
    #print('groundtruth')
    #print('dim : {0}'.format(len(f['groundtruth'][0])))
    #print(f['groundtruth'][0:10])
    # print(type(f['groundtruth'][0][0]))
    # print(f['groundtruth_randomquery'][0:10])
    # print(type(f['groundtruth_randomquery'][0][0]))
    #print('query')
    #print('dim : {0}'.format(len(f['query'][0])))
    # print(f['query'][0:10])
    # print(type(f['query'][0][0]))
    # print(f['randomquery'][0:10])
    # print(type(f['randomquery'][0][0]))

    return f

def to_fvecs(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

def to_ivecs(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('I', int(x))
                fp.write(a)


if __name__ == "__main__":
    #MSong_hdf5_path = "F:/ANNS/DATASET/millionSong.hdf5"
    #enron_hdf5_path = "F:/ANNS/DATASET/enron.hdf5"
    #audio_hdf5_path = "F:/ANNS/DATASET/audio.hdf5"
    
    # f = load_hdf5_file(audio_hdf5_path)
    # to_fvecs('audio_base.fvecs', f['dataset'])
    # print('base done!')
    # to_fvecs('audio_query.fvecs', f['query'])
    # print('query done!')
    # to_ivecs('audio_groundtruth.ivecs', f['groundtruth'])
    # print('groundtruth done!')


    uqv_base_txt_path = "F:/ANNS/DATASET/uqv/UQV_train.txt"
    uqv_query_txt_path = "F:/ANNS/DATASET/uqv/UQV_query.txt"

    #base_data = load_txt_file(uqv_base_txt_path)
    #print("base data len : {0}".format(len(base_data)))
    #to_fvecs('uqv_base.fvecs', base_data)

    query_data = load_txt_file(uqv_query_txt_path)
    print("query data len : {0}".format(len(query_data)))
    to_fvecs('uqv_query.fvecs', query_data)

    print('done!')