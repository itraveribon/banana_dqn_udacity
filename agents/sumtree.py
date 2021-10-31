import numpy as np


class SumTree(object):
    """SumTree implementation based on the code in the following website: https://pylessons.com/CartPole-PER/"""
    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        self.amount_leafs = 0
        self.data_pointer = 0

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        """ tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right
        """
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1
        self.amount_leafs += 1
        self.amount_leafs = min(self.amount_leafs, self.capacity)

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value_to_search):
        parent_index = 0
        followed_path = []
        value_to_search_evolution = []

        while True:
            followed_path.append(parent_index)
            value_to_search_evolution.append(value_to_search)
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= self.tree.shape[0]:
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if value_to_search <= self.tree[left_child_index] or self.tree[right_child_index] == 0:
                    parent_index = left_child_index
                else:
                    value_to_search -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def sum_priorities(self):
        return self.tree[0]  # Returns the root node

    @property
    def max_priority(self):
        return max(self.tree[-self.capacity:])  # Returns the root node

    def __len__(self):
        return self.amount_leafs
