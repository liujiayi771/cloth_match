from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import math
from data_structure import dim_item, dim_fashion_match_sets


def init_dim_item_from_input_line(line, with_tf_idf=False):
    tokens = line.split()
    if not with_tf_idf and len(tokens) == 3:
        return dim_item(tokens[0], tokens[1], tokens[2])
    elif with_tf_idf and len(tokens) == 4:
        return dim_item(tokens[0], tokens[1], tokens[2], tokens[3])
    else:
        print("\nerror line: " + line)
        return None


def init_dim_items_from_filename(filename, with_tf_idf=False):
    items = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n')
            tokens = line.split()
            if not with_tf_idf and len(tokens) == 3:
                items.append(dim_item(tokens[0], tokens[1], tokens[2]))
            if with_tf_idf and len(tokens) == 4:
                items.append(dim_item(tokens[0], tokens[1], tokens[2], tokens[3]))
    return items


def init_item_dict_from_filename(filename, with_tf_idf=False):
    items_dict = {}
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n')
            tokens = line.split()
            if not with_tf_idf and len(tokens) == 3:
                items_dict[tokens[0]] = dim_item(tokens[0], tokens[1], tokens[2])
            if with_tf_idf and len(tokens) == 4:
                items_dict[tokens[0]] = dim_item(tokens[0], tokens[1], tokens[2], tokens[3])
    return items_dict


def init_item_match_dict_from_filename(filename):
    item_match_dict = {}
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n')
            tokens = line.split()
            if len(tokens) == 2:
                item_match_dict[tokens[0]] = tokens[1].split(',')
    return item_match_dict


def init_dim_fashion_match_sets_from_filename(filename):
    match_sets = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n')
            tokens = line.split()
            if len(tokens) == 2:
                match_sets.append(dim_fashion_match_sets(tokens[0], tokens[1]))
    return match_sets


def calculate_tf_idf(single_item, items):
    single_item.tf_idf = {}
    for xt in single_item.terms:
        tf = single_item.count[xt] / len(single_item.terms)
        same_cat_items = filter(lambda x: x.cat_id == single_item.cat_id, items)
        idf = math.log(len(same_cat_items) / len(filter(lambda x: x.count[xt] != 0, same_cat_items)), math.e)
        single_item.tf_idf[xt] = tf * (single_item.alpha * idf + single_item.beta)


def cat_calculate_tf_idf(single_item, items, cat_dict):
    for xt in single_item.terms:
        tf = single_item.count[xt] / len(single_item.terms)
        idf = math.log(len(items) / len(cat_dict[xt]), math.e)
        single_item.tf_idf.append(tf * (single_item.alpha * idf + single_item.beta))


def cat_calculate_tf_idf_without_dict(single_item, items):
    for xt in single_item.terms:
        tf = single_item.count[xt] / len(single_item.terms)
        idf = math.log((len(items) + 1) / (len(filter(lambda x: x.count[xt] != 0, items)) + 1), math.e)
        single_item.tf_idf.append(tf * (single_item.alpha * idf + single_item.beta))


def calculate_similarity(a_item, b_item):
    intersection = list(set(a_item.terms).intersection(b_item.terms))
    res = 0
    for same in intersection:
        res += float(a_item.tf_idf[a_item.terms.index(same)]) * float(b_item.tf_idf[b_item.terms.index(same)])
    return res

