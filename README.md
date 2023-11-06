# Multiclass Text Classification using Scikit-Learn

This repo shows a straightforward pipeline for training a [multi-class](https://en.wikipedia.org/wiki/Multiclass_classification) text classification model using [scikit-learn](https://github.com/scikit-learn/scikit-learn). The training notebook is [here (train_models.ipynb)](./train_models.ipynb). 

This is the kind of training pipeline that I would use at an initial *Proof Of Concept* stage of a project, for example to use in assessing whether Machine Learning is a viable option.

The dataset I've used is the [20 News Groups](http://qwone.com/~jason/20Newsgroups/) text dataset, which consists of forum posts into 20 different newsgroups (topics). The problem being solved is predicting the topic of a given forum post based only on the text content of that post.  

Bearing in mind that some of the categories are unreasonably overlapping (e.g. *soc.religion.christian*, *talk.religion.misc* and *alt.atheism*), here is a confusion matrix of the final model predictions on the test set:

![](/media/confusion_matrix_test.png)

...and here is the final model's achieved accuracy on the test set, under varying confidence requirements:

![](/media/model_accuracy_test.png)

Here is a summary of the training process:

* Training features are created using 2 different methods:

    1. The text is [tokenized](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html) in order to create [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)/[n-gram](https://en.wikipedia.org/wiki/N-gram) features
    
    2. Embedded representations of the documents are created using the google [universal sentence encoder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder) model

* Various different models are compared using k-fold cross validation.

* The final chosen model is trained on the full training data.

* The final model is turned into a scikit-learn [pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline), which contains both the feature preparation and trained model. 

* The final chosen model performance is evaluated on the test set.

Important things that have been omitted:

* Model hyperparameter tuning (all models are just using the [scikit-learn](https://github.com/scikit-learn/scikit-learn) defaults)

* Model explainability (feature importance, partial dependence plots etc.)

* Fine-grained assessment of model performance (e.g. on which subsets of the data does the model perform better/worse)

* Assessing the [probability calibration](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html) of the final classifier model (this is important to do if the model outputs are intended to be interpreted as probabilities).

* Optimization of the parameters of the document tokenization (e.g. stop-word removal, n-gram length, TF-IDF etc.)

# A Note about Jupyter Notebook Rendering

I rendered the jupyter notebook in terminal using 

```bash
jupyter nbconvert --to notebook --execute backup_train_models.ipynb
```