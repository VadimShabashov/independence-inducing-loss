# Description of the presented experiments

The goal of the project is to compare performance of independent losses. 

Initially, the platform was designed for taking each possible configuration of params from `config.yml` file.
However, it turns out, that there are too many experiments to test, and they take too much time to complete.
So, another approach has to be done.

The sequence of experiments in this folder is an attempt to estimate each of the parameters separately. It's not ideal, since
they might behave in a different way when changing together. But it seems like a reasonable approximation.

The experiments descriptions are presented below:
1. Test 1: check that everything is working.
2. Test 2: check that everything is working.
3. Find the best batch size for all models.
4. Find number of epochs and whether classification loss should be used or not.
5. Find the best number of unfrozen layers.
6. Find the best independence loss.
7. Find the best whitening for independence loss.
8. For all datasets, run model with independence loss and the best found parameters.
9. For all datasets, run model with regularization loss or with no specific loss (without regularization and independence loss).

Experiments 9 and 10 allow to compare performance on different datasets of all the models with and without independence loss.
