from __future__ import division
from __future__ import absolute_import
import collections


class dim_item(object):
    """class for items"""
    alpha = 0.008
    beta = 0.07

    def __init__(self, item_id, cat_id, terms, tf_idf=None):
        self.item_id = item_id
        self.cat_id = cat_id
        self.terms = terms.split(',')
        self.count = collections.Counter(self.terms)
        if tf_idf is None:
            self.tf_idf = []
        else:
            self.tf_idf = tf_idf.split(',')

    def __hash__(self):
        return hash(self.cat_id)


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
        match_item = []
        if self.find_item(item):
            for x in self.item_list:
                if item.item_id not in x:
                    match_item += x.split(',')
        return match_item


class user_bought_history(object):
    """class for user bought history"""

    def __init__(self, user_id, item_id, create_at):
        self.key = user_id + create_at
        self.item_id = item_id

