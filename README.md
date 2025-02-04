# Dynaforest

Code for dynaforest, an tree ensemble algorithm with splits decided at the forest level, not the tree. This algorithm was designed to test and understand the effect of correlation among trees in a random forest, and whether the bias of individual trees could be reduced while retaining low correlation between trees. Pseudocode can be found in [this file](pseudocode.pdf), and key formulas and derivatives can be found [here](cumsum.pdf) 

This algorithm is fully coded, and can be ran on your device. After analysis, the key findings are the following:
- On various testing datasets, dynaforest does indeed have lower bias within each tree. However, it also has considerably larger paired tree correlations. This often balances out -- when n_trees is large and so is window, the correlation is often 2x as large as in a general random forest. However, the bias is often small enough that the total error is quite low -- often **significantly** lower than a random forest. 
- Dynaforest's algorithm is necessarily much, much slower than that of random forest (100x or more). AFAICT, there is no algorithm to reduce this time. Parallelization is probably useful, though may be hard to code due to constant communication between parallel calculations. 

TODO:
- Change bootstrapping so it only takes place in the initial trees, use full dataset X for all after
- Change predictions so that it updates correctly
- Add the possibility for initial trees to be multiple layers deep