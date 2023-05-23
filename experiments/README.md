# Description of the presented experiments

The goal of the project is to compare performance of independent losses. 

Initially, the platform was designed for taking each possible configuration of params from `config.yml` file.
However, it turns out, that there are too many experiments to run, and they take too much time to complete.
So, another approach has to be done.

The sequence of experiments in this folder is an attempt to estimate each of the parameters separately. It's not ideal, since
they might behave in a different way when changing together. But it seems like a reasonable approximation.

The experiments descriptions are presented below:
<ul>
<li>No additional losses.</li>
<ol>
<li>Find best batch size for a model without additional losses.</li>
<li>Check whether to use classification loss or not for a model without additional losses.</li>
<li>Find best number of layers to train for a model without additional losses.</li>
</ol>
<li>With independence loss.</li>
<ol start="4">
<li>Find best batch size for a model with independence loss.</li>
<li>Check whether to use classification loss or not for a model with independence losses.</li>
<li>Find best whitening for a model with independence loss.</li>
</ol>
<li>Results.</li>
<ol start="7">
<li>Results for models with decorrelation loss and no loss at all.</li>
<li>Results for models with independence losses.</li>
<li>Results for models with regularization loss.</li>
</ol>
<li>Checking few ideas.</li>
<ol start="10">
<li>Checking large embedding dimension for a model with independence loss.</li>
<li>Check models with no additional losses on a single layer.</li>
<li>Checking a model with independence loss on all layers and no whitening.</li>
</ol>
</ul>
