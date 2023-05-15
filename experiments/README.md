# Description of the presented experiments

The goal of the project is to compare performance of independent losses. 

Initially, the platform was designed for taking each possible configuration of params from `config.yml` file.
However, it turns out, that there are too many experiments to test, and they take too much time to complete.
So, another approach has to be done.

The sequence of experiments in this folder is an attempt to estimate each of the parameters separately. It's not ideal, since
they might behave in a different way when changing together. But it seems like a reasonable approximation.

The experiments descriptions are presented below:
1. Testing part.
   1. Check that everything is working.
   2. Check that everything is working.
2. Best parameters for model without independence loss.
   1. Find the best batch size for each model.
   2. Find whether classification loss should be used or not.
   3. Find the best number of unfrozen layers.
3. Best parameters for model with independence loss.
   1. Run independence losses without whitening for all layers.
   2. Increase constant for independence loss.
   3. Increase batch size.
   4. Add whitening.
   5. Check best number of unfrozen layers.
   6. Find best batch size.
   7. Compare whitening techniques.
   8. Decrease constant for independence loss.
   9. Remove whitening, unfreeze single layer.
   10. Increase constant for independence loss and remove L2.
   11. Return L2, compare embedding dimension and batch size.
   12. Add whitening to the final embeddings.
4. Comparison of the best models on different datasets.
   1. Best models without independence loss (with and without regularization).
   2. Best models with KurtosisLoss or CorrMatLoss.
   3. Best models with NegApproxLoss1 or NegApproxLoss2.
