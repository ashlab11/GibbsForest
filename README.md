# Dynatree

Code for dynatree, an tree ensemble algorithm with splits decided at the forest level, not the tree.


**Remaining Implements:**

* Add in gradient descent -- instead of setting the value of the leaf to be the mean of y, set it to be mean(y) - epsilon * mean(other_predictions)
* Implement max_depth
* Implement min samples

**Pseudocode for regression**

Inputs: X, y, num_trees (N), window (W) max_depth (M), min_samples (S)

* Let: trees be empty array, []
* Let: leafs be a single array,
* Let: tree window be empty queue of max length W
* Let: current prediction be the mean of y, as a singular leaf
* Let: best $e_0$ be current sum squared error
* While len(trees) <= N:
  * For each tree in trees:


Output: trees


$$
E = \sum_{i=1}^n (y_i - o_i - \alpha)^2 \\
\frac{\partial}{\partial \alpha} E = 0 \implies \sum_{i=1}^n y_i - o_i - \alpha = 0 \implies \\
\alpha = \frac1n \sum_{i=1}^n y_i - o_i = \overline{y - o}
$$
