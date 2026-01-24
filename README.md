# My Small PFN

Prior-Data Fitted Networks have recently pushed forward what’s possible on small tabular 
datasets, giving us a powerful new method of analyzing them, a type of data that is dominant 
in many data science fields and has historically been less served by deep learning.

These advancements within data science are mostly driven by PriorLabs with the development 
of [TabPFN](https://github.com/PriorLabs/TabPFN) and more recently Real-TabPFN-2.5, which 
sets state-of-the-art on standard benchmarks of tabular data analysis. Its results after 
a single forward pass with no fine-tuning are competitive with the best ensembles and some 
of their new features make it a very powerful tool for all kinds of data analysts.

Despite the success of their model what really stands out here is not TabPFN-2.5 itself, but 
the idea behind it. Once you have sufficient background building a PFN becomes surprisingly 
accessible, and therefore building one tailored to your data should be an interesting path 
to explore if you really want to get as much as you can out of it.

This project intends to prove this last statement. After two weeks of in-depth analysis of 
TabPFN and its associated papers, I built a small PFN myself that, despite the hardware 
and time limitations, is a strong proof-of-concept with competitive results. It outperforms 
linear regression baselines on almost all tested datasets, and on low data regimes outperforms 
gradient boosted regressors, staying competitive for larger datasets too.

## How to use it

Using my model is quite simple, clone the repository and install the libraries listed on 
`requirements.txt`, then you should be able to run both notebooks, `basic_tests.ipynb` and 
`real_data_tests.ipynb`. 

This repository currently has two different models available, `v1.0` and `v2.0`, details on
their configurations can be found in their respective `my_PFN_config.json` files.

For a simple prediction you can run the following code:

```python
# Import and create the model with the trained weights.
from scripts.my_small_PFN import MyRegressorPFN
model = MyRegressorPFN('v2.0')

# Obtain your own normalized data splits.
X_train, y_train, X_test = my_own_data() # (n_train, features), (n_train,), (n_test, features)

# Fit() the training data in the model.
# This function is just a holder.
model.fit(X_train, y_train)

# Predict() to obtain the output predictions.
# In this case we will take the mean of the predicted distributions.
predicted_y_test = model.predict(X_test, output='mean') # (n_test,)
```

Since this is a proof-of-concept, no heavy preprocessing is performed, this is left to the user. 
Therefore, the PFN expects:

- Normalized features and targets.
- No missing values or *NaN*s.
- `torch.Tensor` or array-like structures that can be converted by `torch.as_tensor()` to floating point tensors.
- Matching shapes (with some freedom), details can be found inside `reshape_concatenate_pre_encoder()`. The ones commented on the code are valid.
- The model outputs `torch.Tensor`. If the model is on inference mode (default) it will output squeezed CPU tensors with no gradient.

For more detailed information on how PFNs work and how to interpret the outputs I strongly 
suggest reading the rest of this file for better understanding.

## What is a PFN?

Imagine we have two sets $X, Y$ and we have a dataset

$$
D_n = (x_i, y_i)_{i=1}^n \text{ with } x_i\in X, y_i \in Y
$$

where $x_i$ will be considered our features and $y_i$ our targets. We also have another feature 
row $x \in X$ and an arbitrary label $y\in Y$, and we want to assess

$$
\mathbb{P}(y \mid x, D_n).
$$

In an ordinary ML mindset, we would provide our inductive bias to solve this particular problem 
by designing a model architecture that we think can mimic the data behavior, then training 
that model on the data and finally running the model on the test rows to obtain our predictions.

PFNs, however, take a different approach, from a Bayesian point of view there are some beliefs we 
hold about the type of data relations that might be present in our dataset. Those beliefs are 
called the **prior**, importantly we can sample models from it to produce datasets 
with exactly the kind of relations we expect to find in our real data. Those datasets will later 
be used to train our model. But how do we build a model that encapsulates this **prior** and our 
inductive bias we are introducing on it?

The following function $\pi$ gives us an optimal solution to our original question. 
Being $\Pi$ our **prior** and $\mathcal{P}$ its support,

$$
\pi(y \mid x, D_n) = \int_{p \in \mathcal{P}} p(y \mid x) \ d\Pi(p \mid D_n).
$$

Under standard assumptions, as $n$ tends to infinity, this function converges to the optimal posterior 
predictive distribution within our **prior**. Therefore, this is the function we will try to mimic.

It turns out we can approximate this function with any model $q_\theta$ that can take as input 
our dataset and $x$ and output a probability distribution over the labels, we do that by minimizing 
negative log likelihood over randomly sampled datasets from our **prior**. 

Then after enough training we will have a functional model that mimics $\pi$, taking training features,
training targets and test features as an input, and outputting us probabilities on all the labels for 
the test targets. The output is determined by our prior, the limitations of our model, and the variance 
in the dataset provided.

This allows us to analyze entire datasets in a single forward pass of the model, with no previous
training on that particular data, performing what is usually called in-context-learning and giving
us a powerful tool for small datasets that does not overfit as long as our **prior** is well designed.

This is just a general overview of the theory and by no means gives a full picture, for a more 
detailed explanation I strongly suggest checking the theory foundations paper cited at the end.

## How does regression work? What does it output?

So far we have described how a PFN classifier works, but how do we turn that into a regressor? The 
answer is quite simple and quite creative at the same time: you turn your regression problem into a 
classification one.

The clearest way to do that is by discretizing the real number line into a finite set of labels. 
Under the assumption that on average the target is normally distributed, then we will split the 
real number line in equally probable buckets under a normal distribution $N(0,1)$, the number of 
buckets will be defined by the model configuration.

That means, following the Bayesian essence of the theory, our regressor will not output a single 
number, but instead it will output a probability distribution of the target over our discretized real 
number line. This does not only solve our regression problem while staying within the theory, but also 
greatly expands the capabilities of our regressor.

This means our model will have different output types as well, if you want a common regressor behavior
you can set `output="mean"` and you will obtain the mean of the discretized distribution, but if 
you instead prefer to see the distribution you can set `output="probs"`, as done in the `basic_tests` 
file, that will give you the probabilities of each bucket, which you can manipulate by using the 
`BucketOps` provided inside the model script. If you also would like to get other values you can set 
`output="values"` and this will return mean, variance and standard deviation of the output 
distribution.

## Code Analysis

Now how does my small PFN actually work? It follows a very similar structure to that of TabPFN-2.5 
but on a smaller scale, let's break it down:

![TabPFN Model Breakdown](https://github.com/PriorLabs/tabpfn-extensions/blob/main/tabpfn_summary.webp)

This image is extracted directly from the [TabPFN Repository](https://github.com/PriorLabs/TabPFN) and 
it summarizes the training we just covered and the forward pass we are about to explain.

All the relevant code can be found inside `scripts/my_small_PFN.py`, this file defines all the torch 
modules needed to do the forward pass as well as some useful functions, let's break it down:

- `ModelConfig`: This dataclass contains all the hyperparameters to define the dimensions of the model, 
as well as some other useful data. It is used to initialize the model, each saved model loads its own 
configuration.

- `BucketOps`: As mentioned before, to be able to do regression our model does not output a value, but 
instead outputs a probability distribution of the target over a discretization of the real number line. 
This class contains multiple functions to use the output probabilities of the model, you can check some 
use cases inside `notebooks/basic_tests.ipynb` or read the descriptions and code of each function for a 
detailed explanation of each one of them.

- `MyRegressorPFN`: This class is the wrapper class for the entire PFN, on initialization it creates 
all the necessary modules to perform the forward pass and loads the weights if specified. The `forward()` 
method is disabled, therefore `fit()` and `predict()` should be used instead, following the same behavior 
as TabPFN-2.5. Some other useful functions are also defined for the class, to see use cases I suggest 
checking the test notebooks.

- **Forward Pass**: All other classes inside the script are parts of the forward pass of the PFN, let's 
    go through it and describe how it works: 

    First we do some minimal preprocessing, making sure the tensors provided are valid, ensuring their shapes 
    match the model expected input and concatenating the feature rows.

    Then we send the tensors to the `EncoderX` and `EncoderY`, these modules will convert our tabular data 
    into a matrix of tokens. Features are joined into small groups and transformed into tokens obtaining a 
    tensor of shape $(B,S,F_g,E)$ where $B$ is the batch size, $S$ is the total number of rows, $F_g$ the 
    amount of feature groups, and $E$ the embedding dimension. In `EncoderY` training targets are concatenated 
    to a vector of zeros representing the test targets, and an additional indicator is attached to signal 
    missing targets. Finally, they are transformed into tokens obtaining a tensor of shape $(B,S,1,E)$.

    After that the two embeddings are concatenated and `AddThinkingTokens` is used to add some extra rows 
    at the beginning, to give the network some more space. This gives us the input for the transformer of 
    shape $(B,T+S,F_g+1,E)$, where $T$ is the number of thinking rows.

    Then the token matrix is sent to the `Transformer`, which every layer performs "double Attention" as can 
    be seen in the picture. This means that first different feature groups within the same row attend to 
    each other, then the tensor is transposed allowing the same feature groups of different rows attend to 
    each other. This basically allows a full attention between all tokens of the matrix, with the only exception 
    that no one attends to test rows during row attention. At the end of every layer a feed forward step is 
    performed and between every step the regular residual addition and layer normalizations are applied.

    After the tensor goes through the entire `Transformer` stack, we send it to the `Decoder`, this module 
    selects the test target tokens and passes them individually through an MLP to obtain the logits for 
    each bucket of each test target, outputting a tensor of shape $(B,test_{size},buckets)$.

    From these logits all the outputs mentioned on the previous section can be computed.

The training files of the model are not available in this repository, for further information about the 
training and my particular **prior** generation you can contact me directly.

## Real Data Tests Analysis

In `notebooks/real_data_tests.ipynb` the `v2.0` PFN has been evaluated on $22$ datasets from different 
domains, sizes, difficulty levels, and structural properties. The datasets have been selected from 
`sklearn` and `OpenML`, with the conditions of being small to medium regression datasets, with no 
missing data and only numerical features, matching the model limitations. 

The models' results have been compared against two different baselines, these being 
`sklearn.linear_model.LinearRegression`, a simple linear regressor, and 
`sklearn.ensemble.HistGradientBoostingRegressor`, a gradient boosting ensemble that serves as a strong 
baseline for our purposes.

For the testing, $10$ random $80/20$ splits are selected, and the data is normalized using the training mean 
and standard deviation. The metrics used are mean squared error and $R^2$ score on the normalized targets. 
These results are fully reproducible by rerunning the notebook with the same random seeds. The results of 
the tests are shown in the following table reporting mean over the 10 splits. To see the full results please
check the notebook file.

| Data Set                          | Rows  | Features | PFN MSE   | LR MSE    | HGBR MSE   |
|:----------------------------------|------:|---------:|:----------|:----------|:-----------|
| Diabetes                          |   442 |       10 | 🟢 0.4544 | 🟡 0.4600 | 🔴 0.5468  |
| California Housing                | 20640 |        8 | 🟡 0.2394 | 🔴 0.3953 | 🟢 0.1689  |
| Longley                           |    16 |        6 | 🟡 0.0189 | 🟢 0.0149 | 🔴 0.9400  |
| CPU Small                         |  8192 |       12 | 🟡 0.1130 | 🔴 0.2784 | 🟢 0.0212  |
| Boston Housing                    |   506 |       13 | 🟡 0.1507 | 🔴 0.2877 | 🟢 0.1230  |
| Concrete Compressive Strength     |  1030 |        8 | 🟡 0.1227 | 🔴 0.3879 | 🟢 0.0872  |
| Airfoil Self-Noise                |  1503 |        5 | 🟡 0.1314 | 🔴 0.4983 | 🟢 0.0897  |
| QSAR Fish Toxicity                |   908 |        6 | 🟡 0.3945 | 🔴 0.4343 | 🟢 0.3837  |
| Naval Propulsion Plant            | 11934 |       14 | 🔴 0.2356 | 🟡 0.2005 | 🟢 0.0126  |
| Naval Propulsion Plant (512 Rows) |   640 |       14 | 🔴 1.0646 | 🟢 0.9093 | 🟡 0.9710  |
| Yacht Hydrodynamics               |   308 |        6 | 🟢 0.0054 | 🔴 0.3969 | 🟡 0.0780  |
| Kin8rm                            |  8192 |        8 | 🟢 0.2570 | 🔴 0.5872 | 🟡 0.2602  |
| Wine Quality Red                  |  1599 |       11 | 🟡 0.6255 | 🔴 0.6543 | 🟢 0.5695  |
| Wine Quality White                |  4898 |       11 | 🟡 0.6423 | 🔴 0.7196 | 🟢 0.5504  |
| Energy Efficiency                 |   768 |        8 | 🟡 0.0219 | 🔴 0.0826 | 🟢 0.0028  |
| Auto MPG                          |   392 |        5 | 🟢 0.1016 | 🔴 0.1737 | 🟡 0.1179  |
| Friedman #1 (64 Rows)             |    64 |       10 | 🟢 0.0545 | 🟡 0.3333 | 🔴 0.3508  |
| Friedman #1 (128 Rows)            |   128 |       10 | 🟢 0.0385 | 🔴 0.2883 | 🟡 0.1970  |
| Friedman #2 (64 Rows)             |    64 |       10 | 🟢 0.0085 | 🟡 0.1670 | 🔴 0.4130  |
| Friedman #2 (128 Rows)            |   128 |       10 | 🟢 0.0022 | 🔴 0.1521 | 🟡 0.0878  |
| Friedman #3 (64 Rows)             |    64 |       10 | 🟢 0.1063 | 🟡 0.5274 | 🔴 0.7573  |
| Friedman #3 (128 Rows)            |   128 |       10 | 🟢 0.0846 | 🔴 0.4870 | 🟡 0.4525  |

The most notable result is that the PFN outperforms HGBR in $10$ out of the $11$ datasets that have a training split 
with $\leq 512$ rows, which is the maximum number of rows the PFN was trained on. This demonstrates a clear generalization 
capability in low-data environments, which is the exact situation PFNs are expected to work on, making clear that 
even with hardware limitations it is possible to create a competitive PFN if you focus on small dataset environments.

We also observe that the PFN is capable of generalizing past its training limitations, showing good results 
on datasets with far more rows than the amount it was trained on. This follows what is argued in the theory 
paper, this being that a transformer based PFN, due to its architecture, can further generalize on bigger 
datasets by decreasing variance.

There is one clear exception to this generalization, this being *Naval Propulsion Plant*, where the PFN performs
far worse than the HGBR despite it being a very learnable deterministic dataset, as seen on the tests. This shows 
a clear limitation of the model despite our previous claim of generalization, but why is that?

As discussed at the end of the paper, despite predictions getting better with more data, the improvements are only 
due to lower variance. The bias though, does not improve, any structure in the data that cannot be learned with 
$\leq 512$ rows will most likely not be learned by our PFN.

The *Naval Propulsion Plant* dataset in particular is a good example of what we just mentioned, if we only allow 
$512$ rows, none of the models manages to find any meaningful structure, as seen in the test. Therefore, our PFN,
due to its training will struggle generalizing despite having a lot of examples, since it will not improve its bias.

Overall, the results of the tests are very promising, dominating on low data regimes and remaining competitive when 
predicting on larger datasets than the ones it was trained on. Despite its clear limitations, since it is just a 
proof-of-concept, it shows the potential of PFNs and their versatility. I hope this inspires other people to create 
their own PFNs and experiment with different kinds of data.

### Other Tests

The `v1.0` model has been tested on preprocessed market data, since it was trained on a heavier noise regime, the 
results are not shown in this repository due to the property of the data. Despite not being able to outperform a 
carefully designed ensemble, it clearly finds signal, performing better than market in almost all time splits and 
across different trained models.

For the fine-tuning of the models, different hyperparameters were tried for the **prior** generation, and limited 
by the models’ size there seems to be a trade-off between different kind of data regimes. If noise is increased 
in the **prior**, noisier datasets will benefit from it, but more deterministic ones will suffer, and vice-versa. 
This is an interesting observation nonetheless, and it shows the importance of designing your **prior** according 
to your objectives.

## License

My-Small-PFN is released under the MIT License. See the LICENSE file for details.

## Contact Information

You can contact me with questions, offers and suggestions at my email:

[miguel.nasarre.budino@gmail.com](mailto:miguel.nasarre.budino@gmail.com)

## Bibliography

For further reading about PFNs and TabPFN-2.5 in particular since their first iteration of TabPFN 
the researchers of this architecture have been publishing multiple reports and papers on the topic. 
For the creation of this project the following ones have been read:

- [2022 — Transformers Can Do Bayesian Inference](https://arxiv.org/abs/2112.10510) (PFN discovery)
- [2023 — TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848) (TabPFN v1)
- [2023 — Statistical Foundations of Prior-Data Fitted Networks](https://proceedings.mlr.press/v202/nagler23a) (Theory foundation of PFNs)
- [2025 — From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models](https://arxiv.org/abs/2501.02945) (Expansion of PFNs to Time Series)
- [2025 — Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) (Nature paper on TabPFN-2.5)
- [2025 — TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models](https://arxiv.org/abs/2511.08667) (TabPFN-2.5 Technical Report)