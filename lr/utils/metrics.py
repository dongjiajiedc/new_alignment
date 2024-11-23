import numpy as np

from utils.mst import reorder
from utils.tree import descendants_traversal, descendants_count



def dasgupta_cost(tree, similarities):
    """ Non-recursive version of DC for binary trees.

    Optimized for speed by reordering similarity matrix for locality
    """
    n = len(list(tree.nodes()))
    root = n - 1
    n_leaves = len(similarities)

    leaves = descendants_traversal(tree)
    n_desc, left_desc = descendants_count(tree)

    cost = [0] * n 

    similarities = reorder(similarities, np.array(leaves), n_leaves)  

    children = [list(tree.neighbors(node)) for node in range(n)]  
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) < 2:
                pass
            elif len(children_) == 2:
                left_c = children_[0]
                right_c = children_[1]

                left_range = [left_desc[left_c], left_desc[left_c] + n_desc[left_c]]
                right_range = [left_desc[right_c], left_desc[right_c] + n_desc[right_c]]
                cost_ = np.add.reduceat(
                    np.add.reduceat(
                        similarities[
                        left_range[0]:left_range[1],
                        right_range[0]:right_range[1]
                        ], [0], axis=1
                    ), [0], axis=0
                )
                cost[node] = cost_[0, 0]

            else:
                assert False, "tree must be binary"
            assert node == stack.pop()

    return 2 * sum(np.array(cost) * np.array(n_desc))
