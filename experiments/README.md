# Description of the presented experiments

The goal of the project is to compare performance of independent losses. 

Initially, the platform was designed for taking each possible configuration of params from `config.yml` file.
However, it turns out, that there are too many experiments to run, and they take too much time to complete.
So, another approach has to be done.

The sequence of experiments in this folder is an attempt to estimate each of the parameters separately. It's not ideal, since
they might behave in a different way when changing together. But it seems like a reasonable approximation.

The experiments descriptions are presented below.

No additional losses.
1. Find best batch size for a model without additional losses.
2. Check whether to use classification loss or not for a model without additional losses.
3. Find best number of layers to train for a model without additional losses.
* With independence loss.

<ol start="10">
<li>foo</li>
</ol>
<ul>
<li>bar</li>
</ul>

  {:start="3"}
  1. Find best batch size for a model with independence loss.
  2. Check whether to use classification loss or not for a model with independence losses.
  3. Find best whitening for a model with independence loss.
* Results.
{:start="7"}
  1. Results for models with decorrelation loss and no loss at all.
  2. Results for models with independence losses.
  3. Results for models with regularization loss.
* Checking few ideas.

{:start="10"}
  1. Checking large embedding dimension for a model with independence loss.
  2. Check models with no additional losses on a single layer.
  3. Checking a model with independence loss on all layers and no whitening.
