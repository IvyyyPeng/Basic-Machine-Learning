import random
from datetime import datetime
from functools import reduce
from collections import defaultdict, Counter
from vector import vector_or, vector_and, distance

NUM_ROWS = NUM_COLS = 32
DATA_TRAINING = 'digit-training.txt'
DATA_TESTING = 'digit-validating.txt'
DATA_PREDICT = 'digit-predicting.txt'

def load_data(p_file):
    global g_data_set
    g_data_set = defaultdict(list)
    with open(p_file, 'r') as data_file:
        while True:
            bits = data_file.read(NUM_ROWS * (NUM_COLS+1))
            if bits == '':
                break
            vector = []
            bits = bits.replace('\n', '')
            for i in bits:
                vector.append(int(i))
            for i in data_file.readline():
                if i.isdigit():
                    digit = int(i)
            g_data_set[digit].append(vector)
    return g_data_set

def show_training_info(p_data_set):
    print('-'*45)
    print('{:>29}'.format('Training Info'))
    print('-'*45)
    total_sample = 0
    for digit in range(0, 10):
        times = len(p_data_set[digit])
        result = '{:>17} = {:>3}'.format(digit, times)
        total_sample += times
        print(result)
    print('-'*45)
    print(('{:>17} = {:>3}'.format('Total Samples', total_sample)))
    print('-'*45)
    print('\n'*3)

def show_testing_info(p_test_good, p_test_bad):
    print('-'*45)
    print('{:>27}'.format('Testing Info'))
    print('-'*45)
    all_ratios = []
    all_good = all_bad = 0
    for digit in range(0, 10):
        good_times = p_test_good[digit]
        bad_times = p_test_bad[digit]
        ratio = good_times / (good_times+bad_times)
        result = ('{:>15} = {:>3}, {:>4}, {:>4.0%}'
                  .format(digit, good_times, bad_times, ratio))
        print(result)
        all_ratios.append(ratio)
        all_good += good_times
        all_bad += bad_times
    total = all_good + all_bad
    accuracy = sum(all_ratios) / len(all_ratios)
    print('-'*45)
    print('{:>15} = {:<5.2%}'.format('Accuracy', accuracy))
    print('{:>15} = {:<5}'.format('Correct/Total',
                                  str(all_good)+'/'+str(total)))
    print('-'*45)

def knn_by_majority(p_nn):
    digits = [digit for dist, digit in p_nn]
    return Counter(digits).most_common(1)[0][0]

def data_by_or():
    global g_data_set
    new_data_set = defaultdict(list)
    for digit in g_data_set:
        v = reduce(vector_or, g_data_set[digit])
        new_data_set[digit] = [v]
    g_data_set = new_data_set
    return g_data_set

def predict_by_knn_dist(p_v1, p_k=7):
    near = []
    for digit in g_data_set:
        for v2 in g_data_set[digit]:
            dist = distance(p_v1, v2)
            near.append((dist, digit))
    near.sort()
    return knn_by_majority(near[:p_k])

def predict_by_knn_or(p_v1, k=1):
    global g_new_data_set
    all_dist = []
    g_new_data_set = data_by_or()
    for digit in g_new_data_set:
        dist = distance(p_v1, g_new_data_set[digit][0])
        all_dist.append((dist, digit))
    all_dist.sort()
    return all_dist[0][1]

def compute_accuracy(p_predict_model, p_file=DATA_TESTING):
    test_bad = defaultdict(int)
    test_good = defaultdict(int)
    with open(p_file, 'r') as data_file:
        while True:
            bits = data_file.read(NUM_ROWS * (NUM_COLS+1))
            if bits == '':
                break
            vector = []
            bits = bits.replace('\n', '')
            for i in bits:
                vector.append(int(i))
            for i in data_file.readline():
                if i.isdigit():
                    digit = int(i)
            ret = p_predict_model(vector)
            test_bad[digit] += ret != digit
            test_good[digit] += ret == digit
        show_training_info(g_data_set)
        show_testing_info(test_good, test_bad)

def predict(p_predict_model, p_file=DATA_PREDICT):
    with open(p_file, 'r') as data_file:
        while True:
            bits = data_file.read(NUM_ROWS * (NUM_COLS+1))
            if bits == '':
                break
            vector = []
            bits = bits.replace('\n', '')
            for i in bits:
                vector.append(int(i))
            data_file.readline()
            ret = p_predict_model(vector)
            print(ret)


def main_knn(p_predict_model=predict_by_knn_dist):
    begin_time = '\nBeginning of Training @ ' + str(datetime.now())[:-7]
    print(begin_time)
    load_data(DATA_TRAINING)
    compute_accuracy(p_predict_model)
    end_time = 'End of Training @ ' + str(datetime.now())[:-7]
    print(end_time, '\n')
    predict(p_predict_model)

main_knn(predict_by_knn_dist)
main_knn(predict_by_knn_or)
