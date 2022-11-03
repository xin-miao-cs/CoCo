"""
Basic operations on graphs.
"""
import numpy as np


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def head_to_tree(head_list):
    """
    Convert a sequence of head indexes into a tree object.
    :param head_list: head indexes (list).
    :return: root of the tree (Tree).
    """
    root = None
    nodes = [Tree() for _ in head_list]

    for i in range(len(nodes)):
        h = head_list[i]
        if h == -1:
            continue
        nodes[i].idx = i
        nodes[i].dist = -1
        if h == 0:
            root = nodes[i]
        else:
            nodes[h-1].add_child(nodes[i])

    assert root is not None
    return root


def tree_to_adj(tree, num_nodes, directed=False, self_loop=False):
    """
    Convert a Tree object to an (numpy) adjacency matrix.
    :param tree: root of the tree (Tree).
    :param num_nodes: number of nodes (int).
    :param directed: directed or not (bool).
    :param self_loop: self loop or not (bool).
    :return: adjacency matrix (ndarray).
    """
    ret = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret
