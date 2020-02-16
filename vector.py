import math as math

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def sum_of_squares(v):
    return sum(v_i * v_i for v_i in v)

def distance(v, w):
    s = vector_subtract(v, w)
    return math.sqrt(sum_of_squares(s))

def vector_or(v, w):
    return [v_i or w_i for v_i, w_i in zip(v, w)]

def vector_and(v, w):
    return [v_i and w_i for v_i, w_i in zip(v, w)]

# def vector_mean(v, w):
