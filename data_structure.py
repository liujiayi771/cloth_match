from __future__ import division
import collections
import math


class dim_item(object):
    """class for items"""
    alpha = 0.008
    beta = 0.07

    def __init__(self, item_id, cat_id, terms, tf_idf=None):
        self.item_id = item_id
        self.cat_id = cat_id
        self.terms = terms.split(',')
        self.count = collections.Counter(self.terms)
        if (tf_idf == None):
            self.tf_idf = []
        else:
            self.tf_idf = tf_idf.split(',')

    def __hash__(self):
        return hash(self.cat_id)

    def calculate_tf_idf(self, items):
        self.tf_idf = {}
        for xt in self.terms:
            tf = self.count[xt] / len(self.terms)
            same_cat_items = filter(lambda x: x.cat_id == self.cat_id, items)
            idf = math.log(len(same_cat_items) / len(filter(lambda x: x.count[xt] != 0, same_cat_items)), math.e)
            self.tf_idf[xt] = tf * (self.alpha * idf + self.beta)

    def cat_calculate_tf_idf(self, items, cat_dict):

        for xt in self.terms:
            tf = self.count[xt] / len(self.terms)
            idf = math.log(len(items) / len(cat_dict[xt]), math.e)
            self.tf_idf.append(tf * (self.alpha * idf + self.beta))

    def cat_calculate_tf_idf_without_dict(self, items):

        for xt in self.terms:
            tf = self.count[xt] / len(self.terms)
            idf = math.log((len(items) + 1) / (len(filter(lambda x: x.count[xt] != 0, items)) + 1), math.e)
            self.tf_idf.append(tf * (self.alpha * idf + self.beta))

    def similarity(self, other_item):
        intersection = list(set(self.terms).intersection(other_item.terms))
        res = 0
        for same in intersection:
            res += float(self.tf_idf[self.terms.index(same)]) * float(other_item.tf_idf[other_item.terms.index(same)])
        return res


class dim_fashion_match_sets(object):
    """class for matchsets"""

    def __init__(self, coll_id, item_list):
        self.coll_id = coll_id
        self.item_list = item_list.split(';')

    def find_item(self, item):
        find = False
        for x in self.item_list:
            if item.item_id in x:
                find = True
        return find

    def get_match_item(self, item):
        match_item = set()
        if self.find_item(item):
            for x in self.item_list:
                if item.item_id not in x:
                    match_item = match_item.union(set(x.split(',')))
        return match_item
