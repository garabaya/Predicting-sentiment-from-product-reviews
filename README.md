# Predicting sentiment from product reviews


The goal of this first notebook is to explore logistic regression and feature engineering with existing Turi Create functions.

In this notebook you will use product review data from Amazon.com to predict whether the sentiments about a product (from its reviews) are positive or negative.

* Use SFrames to do some feature engineering
* Train a logistic regression model to predict the sentiment of product reviews.
* Inspect the weights (coefficients) of a trained logistic regression model.
* Make a prediction (both class and probability) of sentiment for a new product review.
* Given the logistic regression weights, predictors and ground truth labels, write a function to compute the **accuracy** of the model.
* Inspect the coefficients of the logistic regression model and interpret their meanings.
* Compare multiple logistic regression models.

Let's get started!
    
## Fire up Turi Create

Make sure you have the latest version of Turi Create.


```python
from __future__ import division
import turicreate
import math
import string
```

# Data preparation

We will use a dataset consisting of baby product reviews on Amazon.com.


```python
products = turicreate.SFrame('amazon_baby.sframe/')
```

Now, let us see a preview of what the dataset looks like.


```python
products
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Flannel Wipes</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">These flannel wipes are<br>OK, but in my opinion ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Wipe Pouch</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">it came early and was not<br>disappointed. i love ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Annas Dream Full Quilt<br>with 2 Shams ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Very soft and comfortable<br>and warmer than it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This is a product well<br>worth the purchase.  I ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">All of my kids have cried<br>non-stop when I tried to ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">When the Binky Fairy came<br>to our house, we didn&#x27;t ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A Tale of Baby&#x27;s Days<br>with Peter Rabbit ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lovely book, it&#x27;s bound<br>tightly so you may no ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Perfect for new parents.<br>We were able to keep ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A friend of mine pinned<br>this product on Pinte ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This has been an easy way<br>for my nanny to record ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
    </tr>
</table>
[183531 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



## Build the word count vector for each review

Let us explore a specific example of a baby product.



```python
products[269]
```




    {'name': 'The First Years Massaging Action Teether',
     'review': 'A favorite in our house!',
     'rating': 5.0}



Now, we will perform 2 simple data transformations:

1. Remove punctuation using [Python's built-in](https://docs.python.org/2/library/string.html) string functionality.
2. Transform the reviews into word-counts.

**Aside**. In this notebook, we remove all punctuations for the sake of simplicity. A smarter approach to punctuations would preserve phrases such as "I'd", "would've", "hadn't" and so forth. See [this page](https://www.cis.upenn.edu/~treebank/tokenization.html) for an example of smart handling of punctuations.


```python
import string 
def remove_punctuation(text):
    try: # python 2.x
        text = text.translate(None, string.punctuation) 
    except: # python 3.x
        translator = text.maketrans('', '', string.punctuation)
        text = text.translate(translator)
    return text

review_without_punctuation = products['review'].apply(remove_punctuation)
products['word_count'] = turicreate.text_analytics.count_words(review_without_punctuation)
```

Now, let us explore what the sample example above looks like after these 2 transformations. Here, each entry in the **word_count** column is a dictionary where the key is the word and the value is a count of the number of times the word occurs.


```python
products[269]['word_count']
```




    {'our': 1.0, 'in': 1.0, 'favorite': 1.0, 'house': 1.0, 'a': 1.0}



## Extract sentiments

We will **ignore** all reviews with *rating = 3*, since they tend to have a neutral sentiment.


```python
products = products[products['rating'] != 3]
len(products)
```




    166752



Now, we will assign reviews with a rating of 4 or higher to be *positive* reviews, while the ones with rating of 2 or lower are *negative*. For the sentiment column, we use +1 for the positive class label and -1 for the negative class label.


```python
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sentiment</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Wipe Pouch</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">it came early and was not<br>disappointed. i love ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;recommend&#x27;: 1.0,<br>&#x27;highly&#x27;: 1.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Annas Dream Full Quilt<br>with 2 Shams ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Very soft and comfortable<br>and warmer than it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;quilt&#x27;: 1.0, &#x27;this&#x27;:<br>1.0, &#x27;for&#x27;: 1.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This is a product well<br>worth the purchase.  I ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;tool&#x27;: 1.0, &#x27;clever&#x27;:<br>1.0, &#x27;approach&#x27;: 2.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">All of my kids have cried<br>non-stop when I tried to ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;rock&#x27;: 1.0,<br>&#x27;headachesthanks&#x27;: 1.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">When the Binky Fairy came<br>to our house, we didn&#x27;t ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;thumb&#x27;: 1.0, &#x27;or&#x27;: 1.0,<br>&#x27;break&#x27;: 1.0, &#x27;trying&#x27;: ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A Tale of Baby&#x27;s Days<br>with Peter Rabbit ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lovely book, it&#x27;s bound<br>tightly so you may no ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;2995&#x27;: 1.0, &#x27;for&#x27;: 1.0,<br>&#x27;barnes&#x27;: 1.0, &#x27;at&#x27;:  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Perfect for new parents.<br>We were able to keep ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;right&#x27;: 1.0, &#x27;because&#x27;:<br>1.0, &#x27;questions&#x27;: 1.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A friend of mine pinned<br>this product on Pinte ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;like&#x27;: 1.0, &#x27;and&#x27;: 1.0,<br>&#x27;changes&#x27;: 1.0, &#x27;the&#x27;: ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This has been an easy way<br>for my nanny to record ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;in&#x27;: 1.0, &#x27;pages&#x27;: 1.0,<br>&#x27;out&#x27;: 1.0, &#x27;run&#x27;: 1.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I love this journal and<br>our nanny uses it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;tracker&#x27;: 1.0, &#x27;now&#x27;:<br>1.0, &#x27;postits&#x27;: 1.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
[166752 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Now, we can see that the dataset contains an extra column called **sentiment** which is either positive (+1) or negative (-1).

## Split data into training and test sets

Let's perform a train/test split with 80% of the data in the training set and 20% of the data in the test set. We use `seed=1` so that everyone gets the same result.


```python
train_data, test_data = products.random_split(.8, seed=1)
print(len(train_data))
print(len(test_data))
```

    133416
    33336


# Train a sentiment classifier with logistic regression

We will now use logistic regression to create a sentiment classifier on the training data. This model will use the column **word_count** as a feature and the column **sentiment** as the target. We will use `validation_set=None` to obtain same results as everyone else.

**Note:** This line may take 1-2 minutes.


```python
sentiment_model = turicreate.logistic_classifier.create(train_data,
                                                        target = 'sentiment',
                                                        features=['word_count'],
                                                        validation_set=None)
```


<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 133416</pre>



<pre>Number of classes           : 2</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 121712</pre>



<pre>Number of coefficients      : 121713</pre>



<pre>Starting L-BFGS</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+</pre>



<pre>| Iteration | Passes   | Step size | Elapsed Time | Training Accuracy |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+</pre>



<pre>| 0         | 4        | 0.250000  | 1.524153     | 0.840754          |</pre>



<pre>| 1         | 9        | 3.250000  | 2.101038     | 0.941514          |</pre>



<pre>| 2         | 11       | 2.778177  | 2.385973     | 0.942638          |</pre>



<pre>| 3         | 12       | 2.778177  | 2.565347     | 0.967822          |</pre>



<pre>| 4         | 13       | 2.778177  | 2.764583     | 0.976495          |</pre>



<pre>| 5         | 14       | 2.778177  | 2.951252     | 0.976495          |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+</pre>



```python
sentiment_model
```




    Class                          : LogisticClassifier
    
    Schema
    ------
    Number of coefficients         : 121713
    Number of examples             : 133416
    Number of classes              : 2
    Number of feature columns      : 1
    Number of unpacked features    : 121712
    
    Hyperparameters
    ---------------
    L1 penalty                     : 0.0
    L2 penalty                     : 0.01
    
    Training Summary
    ----------------
    Solver                         : lbfgs
    Solver iterations              : 5
    Solver status                  : TERMINATED: Terminated due to numerical difficulties.
    Training time (sec)            : 1.5535
    
    Settings
    --------
    Log-likelihood                 : inf
    
    Highest Positive Coefficients
    -----------------------------
    word_count[offsi]              : 21.7657
    word_count[kidsgood]           : 21.6818
    word_count[conclusions]        : 21.6818
    word_count[easycheap]          : 21.6818
    word_count[torsional]          : 21.6818
    
    Lowest Negative Coefficients
    ----------------------------
    word_count[wahwah]             : -21.8302
    word_count[timesopros]         : -21.8302
    word_count[pumpabout]          : -21.8302
    word_count[lactinai]           : -21.8302
    word_count[nonclogged]         : -21.8302



**Aside**. You may get a warning to the effect of "Terminated due to numerical difficulties --- this model may not be ideal". It means that the quality metric (to be covered in Module 3) failed to improve in the last iteration of the run. The difficulty arises as the sentiment model puts too much weight on extremely rare words. A way to rectify this is to apply regularization, to be covered in Module 4. Regularization lessens the effect of extremely rare words. For the purpose of this assignment, however, please proceed with the model above.

Now that we have fitted the model, we can extract the weights (coefficients) as an SFrame as follows:


```python
weights = sentiment_model.coefficients
weights.column_names()
```




    ['name', 'index', 'class', 'value', 'stderr']



There are a total of `121713` coefficients in the model. Recall from the lecture that positive weights $w_j$ correspond to weights that cause positive sentiment, while negative weights correspond to negative sentiment. 

Fill in the following block of code to calculate how many *weights* are positive ( >= 0). (**Hint**: The `'value'` column in SFrame *weights* must be positive ( >= 0)).


```python
num_positive_weights = len(weights[weights['value'] >= 0])
num_negative_weights = len(weights) - num_positive_weights

print("Number of positive weights: %s " % num_positive_weights)
print("Number of negative weights: %s " % num_negative_weights)
```

    Number of positive weights: 91073 
    Number of negative weights: 30640 


**Quiz Question:** How many weights are >= 0?

## Making predictions with logistic regression

Now that a model is trained, we can make predictions on the **test data**. In this section, we will explore this in the context of 3 examples in the test dataset.  We refer to this set of 3 examples as the **sample_test_data**.


```python
sample_test_data = test_data[10:13]
print(sample_test_data['rating'])
sample_test_data
```

    [5.0, 2.0, 1.0]





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sentiment</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Our Baby Girl Memory Book</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Absolutely love it and<br>all of the Scripture in ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;again&#x27;: 1.0, &#x27;book&#x27;:<br>1.0, &#x27;same&#x27;: 1.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Wall Decor Removable<br>Decal Sticker - Colorful ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Would not purchase again<br>or recommend. The decals ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;peeling&#x27;: 1.0, &#x27;5&#x27;:<br>1.0, &#x27;about&#x27;: 1.0, &#x27;f ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">New Style Trailing Cherry<br>Blossom Tree Decal ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Was so excited to get<br>this product for my baby ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{&#x27;on&#x27;: 1.0, &#x27;waste&#x27;: 1.0,<br>&#x27;wouldnt&#x27;: 1.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
    </tr>
</table>
[3 rows x 5 columns]<br/>
</div>



Let's dig deeper into the first row of the **sample_test_data**. Here's the full review:


```python
sample_test_data[0]['review']
```




    'Absolutely love it and all of the Scripture in it.  I purchased the Baby Boy version for my grandson when he was born and my daughter-in-law was thrilled to receive the same book again.'



That review seems pretty positive.

Now, let's see what the next row of the **sample_test_data** looks like. As we could guess from the sentiment (-1), the review is quite negative.


```python
sample_test_data[1]['review']
```




    'Would not purchase again or recommend. The decals were thick almost plastic like and were coming off the wall as I was applying them! The would NOT stick! Literally stayed stuck for about 5 minutes then started peeling off.'



We will now make a **class** prediction for the **sample_test_data**. The `sentiment_model` should predict **+1** if the sentiment is positive and **-1** if the sentiment is negative. Recall from the lecture that the **score** (sometimes called **margin**) for the logistic regression model  is defined as:

$$
\mbox{score}_i = \mathbf{w}^T h(\mathbf{x}_i)
$$ 

where $h(\mathbf{x}_i)$ represents the features for example $i$.  We will write some code to obtain the **scores** using Turi Create. For each row, the **score** (or margin) is a number in the range **[-inf, inf]**.


```python
scores = sentiment_model.predict(sample_test_data, output_type='margin')
print(scores)
```

    [4.78890730921402, -3.000782222462476, -8.188501360762789]


### Predicting sentiment

These scores can be used to make class predictions as follows:

$$
\hat{y} = 
\left\{
\begin{array}{ll}
      +1 & \mathbf{w}^T h(\mathbf{x}_i) > 0 \\
      -1 & \mathbf{w}^T h(\mathbf{x}_i) \leq 0 \\
\end{array} 
\right.
$$

Using scores, write code to calculate $\hat{y}$, the class predictions:


```python
predictions = []
for score in scores:
    if score > 0:
        predictions.append(1)
    else:
        predictions.append(-1)
predictions
```




    [1, -1, -1]



Run the following code to verify that the class predictions obtained by your calculations are the same as that obtained from Turi Create.


```python
print("Class predictions according to Turi Create:")
print(sentiment_model.predict(sample_test_data))
```

    Class predictions according to Turi Create:
    [1, -1, -1]


**Checkpoint**: Make sure your class predictions match with the one obtained from Turi Create.

### Probability predictions

Recall from the lectures that we can also calculate the probability predictions from the scores using:
$$
P(y_i = +1 | \mathbf{x}_i,\mathbf{w}) = \frac{1}{1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))}.
$$

Using the variable **scores** calculated previously, write code to calculate the probability that a sentiment is positive using the above formula. For each row, the probabilities should be a number in the range **[0, 1]**.


```python
probabilities = []
for score in scores:
    probability = 1 / (1 + math.exp(-score))
    probabilities.append(probability)
probabilities
```




    [0.9917471313286885, 0.0473905474871234, 0.00027775277121725535]



**Checkpoint**: Make sure your probability predictions match the ones obtained from Turi Create.


```python
print("Class predictions according to Turi Create:")
print(sentiment_model.predict(sample_test_data, output_type='probability'))
```

    Class predictions according to Turi Create:
    [0.9917471313286885, 0.04739054748712339, 0.0002777527712172554]


**Quiz Question:** Of the three data points in **sample_test_data**, which one (first, second, or third) has the **lowest probability** of being classified as a positive review?

# Find the most positive (and negative) review

We now turn to examining the full test dataset, **test_data**, and use Turi Create to form predictions on all of the test data points for faster performance.

Using the `sentiment_model`, find the 20 reviews in the entire **test_data** with the **highest probability** of being classified as a **positive review**. We refer to these as the "most positive reviews."

To calculate these top-20 reviews, use the following steps:
1.  Make probability predictions on **test_data** using the `sentiment_model`. (**Hint:** When you call `.predict` to make predictions on the test data, use option `output_type='probability'` to output the probability rather than just the most likely class.)
2.  Sort the data according to those predictions and pick the top 20. (**Hint:** You can use the `.topk` method on an SFrame to find the top k rows sorted according to the value of a specified column.)


```python
test_data['probability'] = sentiment_model.predict(test_data, output_type='probability')
test_data.topk('probability', k=20).print_rows(num_rows=20)
```

    +-------------------------------+-------------------------------+--------+
    |              name             |             review            | rating |
    +-------------------------------+-------------------------------+--------+
    | Fisher-Price Cradle 'N Swi... | My husband and I cannot st... |  5.0   |
    | The Original CJ's BuTTer (... | I'm going to try to review... |  4.0   |
    | Baby Jogger City Mini GT D... | We are well pleased with t... |  4.0   |
    | Diono RadianRXT Convertibl... | Like so many others before... |  5.0   |
    | Diono RadianRXT Convertibl... | I bought this seat for my ... |  5.0   |
    | Graco Pack 'n Play Element... | My husband and I assembled... |  4.0   |
    | Maxi-Cosi Pria 70 with Tin... | We love this car seat!! It... |  5.0   |
    | Britax 2012 B-Agile Stroll... | [I got this stroller for m... |  4.0   |
    | Quinny 2012 Buzz Stroller,... | Choice - Quinny Buzz 2011 ... |  4.0   |
    | Roan Rocco Classic Pram St... | Great Pram Rocco!!!!!!I bo... |  5.0   |
    | Britax Decathlon Convertib... | I researched a few differe... |  4.0   |
    | bumGenius One-Size Snap Cl... | Warning: long review!Short... |  5.0   |
    | Infantino Wrap and Tie Bab... | I bought this carrier when... |  5.0   |
    | Baby Einstein Around The W... | I am so HAPPY I brought th... |  5.0   |
    | Britax Frontier Booster Ca... | My family previously owned... |  5.0   |
    | Evenflo X Sport Plus Conve... | After seeing this in Paren... |  5.0   |
    | P'Kolino Silly Soft Seatin... | I've purchased both the P'... |  4.0   |
    | Peg Perego Aria Light Weig... | We have 3 strollers...one ... |  5.0   |
    | Fisher-Price Rainforest Me... | My daughter wasn't able to... |  5.0   |
    | Lilly Gold Sit 'n' Stroll ... | I just completed a two-mon... |  5.0   |
    +-------------------------------+-------------------------------+--------+
    +-------------------------------+-----------+-------------+
    |           word_count          | sentiment | probability |
    +-------------------------------+-----------+-------------+
    | {'recommendations': 1.0, '... |     1     |     1.0     |
    | {'order': 1.0, 'latest': 1... |     1     |     1.0     |
    | {'buy': 1.0, 'deal': 1.0, ... |     1     |     1.0     |
    | {'stroller': 1.0, 'traveli... |     1     |     1.0     |
    | {'really': 1.0, 'needs': 1... |     1     |     1.0     |
    | {'works': 1.0, 'sleeping':... |     1     |     1.0     |
    | {'well': 1.0, 'knowing': 1... |     1     |     1.0     |
    | {'allaround': 1.0, 'bill':... |     1     |     1.0     |
    | {'shop': 1.0, 'lucked': 1.... |     1     |     1.0     |
    | {'sell': 1.0, 'regret': 1.... |     1     |     1.0     |
    | {'enough': 1.0, 'big': 1.0... |     1     |     1.0     |
    | {'section': 1.0, 'comment'... |     1     |     1.0     |
    | {'well': 1.0, 'mouth': 1.0... |     1     |     1.0     |
    | {'go': 1.0, 'steals': 1.0,... |     1     |     1.0     |
    | {'looks': 1.0, 'now': 1.0,... |     1     |     1.0     |
    | {'compliments': 1.0, 'gett... |     1     |     1.0     |
    | {'admit': 1.0, 'though': 1... |     1     |     1.0     |
    | {'collapses': 1.0, 'awesom... |     1     |     1.0     |
    | {'youll': 1.0, 'afford': 1... |     1     |     1.0     |
    | {'toddler': 1.0, 'or': 1.0... |     1     |     1.0     |
    +-------------------------------+-----------+-------------+
    [20 rows x 6 columns]
    


**Quiz Question**: Which of the following products are represented in the 20 most positive reviews? [multiple choice]


Now, let us repeat this exercise to find the "most negative reviews." Use the prediction probabilities to find the  20 reviews in the **test_data** with the **lowest probability** of being classified as a **positive review**. Repeat the same steps above but make sure you **sort in the opposite order**.


```python
test_data.topk('probability', k=20, reverse=True).print_rows(num_rows=20)
```

    +-------------------------------+-------------------------------+--------+
    |              name             |             review            | rating |
    +-------------------------------+-------------------------------+--------+
    | Luna Lullaby Bosom Baby Nu... | I have the boppy deluxe pi... |  5.0   |
    | The First Years True Choic... | Note: we never installed b... |  1.0   |
    | Jolly Jumper Arctic Sneak ... | I am a "research-aholic" i... |  5.0   |
    | Motorola MBP36 Remote Wire... | I could go on and on about... |  4.0   |
    | VTech Communications Safe ... | This is my second video mo... |  1.0   |
    | Fisher-Price Ocean Wonders... | We have not had ANY luck w... |  2.0   |
    | Levana Safe N'See Digital ... | This is the first review I... |  1.0   |
    | Safety 1st High-Def Digita... | We bought this baby monito... |  1.0   |
    | Snuza Portable Baby Moveme... | I would have given the pro... |  1.0   |
    | Adiri BPA Free Natural Nur... | I will try to write an obj... |  2.0   |
    | Samsung SEW-3037W Wireless... | Reviewers. You failed me!T... |  1.0   |
    | Motorola Digital Video Bab... | DO NOT BUY THIS BABY MONIT... |  1.0   |
    | Cloth Diaper Sprayer--styl... | I bought this sprayer out ... |  1.0   |
    | Munchkin Nursery Projector... | Updated January 3, 2014.  ... |  1.0   |
    | VTech Communications Safe ... | First, the distance on the... |  1.0   |
    | Carter's Monkey Bars Music... | While the product was ship... |  2.0   |
    | Safety 1st Lift Lock and S... | Don't buy this product. If... |  1.0   |
    | MyLine Eco baby play mat-A... | When i got this mat,  i wa... |  1.0   |
    |    Fisher-Price Royal Potty   | This was the worst potty e... |  1.0   |
    | Evenflo Take Me Too Premie... | I am absolutely disgusted ... |  1.0   |
    +-------------------------------+-------------------------------+--------+
    +-------------------------------+-----------+------------------------+
    |           word_count          | sentiment |      probability       |
    +-------------------------------+-----------+------------------------+
    | {'leacho': 1.0, 'pregnant'... |     1     | 3.229790842409583e-63  |
    | {'least': 1.0, 'wont': 1.0... |     -1    | 1.632282318647957e-24  |
    | {'worthless': 1.0, 'wastin... |     1     | 8.110311382112413e-20  |
    | {'best': 1.0, 'definitely'... |     1     | 7.797281605698468e-16  |
    | {'thisbuyer': 1.0, 'creeki... |     -1    | 1.8411614798627783e-14 |
    | {'market': 1.0, 'same': 1.... |     -1    | 6.345094941886445e-14  |
    | {'anything': 1.0, 'post': ... |     -1    | 6.578528513081438e-14  |
    | {'looking': 1.0, 'recommen... |     -1    | 1.540841001189673e-13  |
    | {'from': 1.0, 'restaurant'... |     -1    | 6.301835289179802e-13  |
    | {'buy': 1.0, 'refuses': 1.... |     -1    | 9.314560905794037e-13  |
    | {'between': 1.0, 'easy': 1... |     -1    | 5.920602970145504e-12  |
    | {'around': 1.0, 'will': 1.... |     -1    | 5.986200850007472e-12  |
    | {'poopy': 1.0, 'else': 1.0... |     -1    | 1.0091402072240517e-11 |
    | {'grr': 1.0, 'whole': 1.0,... |     -1    | 3.455183274564835e-11  |
    | {'like': 1.0, 'productsdo'... |     -1    | 4.4328017170103046e-11 |
    | {'disappointed': 1.0, 'unh... |     -1    | 4.669455335890692e-11  |
    | {'brand': 1.0, 'better': 1... |     -1    | 1.0383337454022507e-10 |
    | {'damaged': 1.0, 'if': 1.0... |     -1    | 2.0250237927224839e-10 |
    | {'fisher': 1.0, 'keep': 1.... |     -1    | 3.935257304117865e-10  |
    | {'purchases': 1.0, 'defini... |     -1    | 4.829851675093441e-10  |
    +-------------------------------+-----------+------------------------+
    [20 rows x 6 columns]
    


**Quiz Question**: Which of the following products are represented in the 20 most negative reviews?  [multiple choice]

## Compute accuracy of the classifier

We will now evaluate the accuracy of the trained classifier. Recall that the accuracy is given by


$$
\mbox{accuracy} = \frac{\mbox{# correctly classified examples}}{\mbox{# total examples}}
$$

This can be computed as follows:

* **Step 1:** Use the trained model to compute class predictions (**Hint:** Use the `predict` method)
* **Step 2:** Count the number of data points when the predicted class labels match the ground truth labels (called `true_labels` below).
* **Step 3:** Divide the total number of correct predictions by the total number of data points in the dataset.

Complete the function below to compute the classification accuracy:


```python
def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    ## YOUR CODE HERE
    data['prediction'] = model.predict(data)
    data['true_label'] = true_labels
    # Compute the number of correctly classified examples
    ## YOUR CODE HERE
    num_correct = 0
    for i in data:
        if i['prediction'] == i['true_label']:
            num_correct +=1

    # Then compute accuracy by dividing num_correct by total number of examples
    ## YOUR CODE HERE
    accuracy = num_correct / len(data)
    
    return accuracy
```

Now, let's compute the classification accuracy of the **sentiment_model** on the **test_data**.


```python
get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
```




    0.9221862251019919



**Quiz Question**: What is the accuracy of the **sentiment_model** on the **test_data**? Round your answer to 2 decimal places (e.g. 0.76).

**Quiz Question**: Does a higher accuracy value on the **training_data** always imply that the classifier is better?

## Learn another classifier with fewer words

There were a lot of words in the model we trained above. We will now train a simpler logistic regression model using only a subset of words that occur in the reviews. For this assignment, we selected a 20 words to work with. These are:


```python
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
```


```python
len(significant_words)
```




    20



For each review, we will use the **word_count** column and trim out all words that are **not** in the **significant_words** list above. We will use the [SArray dictionary trim by keys functionality]( https://dato.com/products/create/docs/generated/graphlab.SArray.dict_trim_by_keys.html). Note that we are performing this on both the training and test set.


```python
train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
```

Let's see what the first example of the dataset looks like:


```python
train_data[0]['review']
```




    'it came early and was not disappointed. i love planet wise bags and now my wipe holder. it keps my osocozy wipes moist and does not leak. highly recommend it.'



The **word_count** column had been working with before looks like the following:


```python
print(train_data[0]['word_count'])
```

    {'recommend': 1.0, 'highly': 1.0, 'disappointed': 1.0, 'love': 1.0, 'it': 3.0, 'planet': 1.0, 'and': 3.0, 'bags': 1.0, 'wipes': 1.0, 'not': 2.0, 'early': 1.0, 'came': 1.0, 'i': 1.0, 'does': 1.0, 'wise': 1.0, 'my': 2.0, 'was': 1.0, 'now': 1.0, 'wipe': 1.0, 'holder': 1.0, 'leak': 1.0, 'keps': 1.0, 'osocozy': 1.0, 'moist': 1.0}


Since we are only working with a subset of these words, the column **word_count_subset** is a subset of the above dictionary. In this example, only 2 `significant words` are present in this review.


```python
print(train_data[0]['word_count_subset'])
```

    {'disappointed': 1.0, 'love': 1.0}


## Train a logistic regression model on a subset of data

We will now build a classifier with **word_count_subset** as the feature and **sentiment** as the target. 


```python
simple_model = turicreate.logistic_classifier.create(train_data,
                                                     target = 'sentiment',
                                                     features=['word_count_subset'],
                                                     validation_set=None)
simple_model
```


<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 133416</pre>



<pre>Number of classes           : 2</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 20</pre>



<pre>Number of coefficients      : 21</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+-------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Accuracy |</pre>



<pre>+-----------+----------+--------------+-------------------+</pre>



<pre>| 1         | 2        | 0.107316     | 0.862917          |</pre>



<pre>| 2         | 3        | 0.157798     | 0.865713          |</pre>



<pre>| 3         | 4        | 0.212629     | 0.866478          |</pre>



<pre>| 4         | 5        | 0.260994     | 0.866748          |</pre>



<pre>| 5         | 6        | 0.312714     | 0.866815          |</pre>



<pre>| 6         | 7        | 0.360144     | 0.866815          |</pre>



<pre>+-----------+----------+--------------+-------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    Class                          : LogisticClassifier
    
    Schema
    ------
    Number of coefficients         : 21
    Number of examples             : 133416
    Number of classes              : 2
    Number of feature columns      : 1
    Number of unpacked features    : 20
    
    Hyperparameters
    ---------------
    L1 penalty                     : 0.0
    L2 penalty                     : 0.01
    
    Training Summary
    ----------------
    Solver                         : newton
    Solver iterations              : 6
    Solver status                  : SUCCESS: Optimal solution found.
    Training time (sec)            : 0.3685
    
    Settings
    --------
    Log-likelihood                 : 44323.7254
    
    Highest Positive Coefficients
    -----------------------------
    word_count_subset[loves]       : 1.6773
    word_count_subset[perfect]     : 1.5145
    word_count_subset[love]        : 1.3654
    (intercept)                    : 1.2995
    word_count_subset[easy]        : 1.1937
    
    Lowest Negative Coefficients
    ----------------------------
    word_count_subset[disappointed] : -2.3551
    word_count_subset[return]      : -2.1173
    word_count_subset[waste]       : -2.0428
    word_count_subset[broke]       : -1.658
    word_count_subset[money]       : -0.8979



We can compute the classification accuracy using the `get_classification_accuracy` function you implemented earlier.


```python
get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
```




    0.8693004559635229



Now, we will inspect the weights (coefficients) of the **simple_model**:


```python
simple_model.coefficients
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">class</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2995449552027043</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.012088854133053259</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">disappointed</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2.3550925006107253</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.050414988855697916</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">love</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.3654354936790372</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.03035462951090517</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">well</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.5042567463979284</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.021381300630990033</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">product</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.320555492995575</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.015431132136201635</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">loves</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.6772714555592918</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.04823282753835012</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">little</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.5206286360250184</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.021469147566490373</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">work</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.6217000124253143</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.023033059794584827</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">easy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.1936618983284648</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.029288869202029586</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">great</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.9446912694798443</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.020950992659050018</td>
    </tr>
</table>
[21 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Let's sort the coefficients (in descending order) by the **value** to obtain the coefficients with the most positive effect on the sentiment.


```python
simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)
```

    +-------------------+--------------+-------+----------------------+
    |        name       |    index     | class |        value         |
    +-------------------+--------------+-------+----------------------+
    | word_count_subset |    loves     |   1   |  1.6772714555592918  |
    | word_count_subset |   perfect    |   1   |  1.5144862670271348  |
    | word_count_subset |     love     |   1   |  1.3654354936790372  |
    |    (intercept)    |     None     |   1   |  1.2995449552027043  |
    | word_count_subset |     easy     |   1   |  1.1936618983284648  |
    | word_count_subset |    great     |   1   |  0.9446912694798443  |
    | word_count_subset |    little    |   1   |  0.5206286360250184  |
    | word_count_subset |     well     |   1   |  0.5042567463979284  |
    | word_count_subset |     able     |   1   |  0.1914383022947509  |
    | word_count_subset |     old      |   1   |  0.0853961886678159  |
    | word_count_subset |     car      |   1   | 0.05883499006802042  |
    | word_count_subset |     less     |   1   | -0.20970981521595644 |
    | word_count_subset |   product    |   1   |  -0.320555492995575  |
    | word_count_subset |    would     |   1   | -0.3623089477110012  |
    | word_count_subset |     even     |   1   |  -0.511738551270056  |
    | word_count_subset |     work     |   1   | -0.6217000124253143  |
    | word_count_subset |    money     |   1   | -0.8978841557762813  |
    | word_count_subset |    broke     |   1   | -1.6579644783802772  |
    | word_count_subset |    waste     |   1   | -2.0427736110037236  |
    | word_count_subset |    return    |   1   |  -2.117296597184635  |
    | word_count_subset | disappointed |   1   | -2.3550925006107253  |
    +-------------------+--------------+-------+----------------------+
    +----------------------+
    |        stderr        |
    +----------------------+
    | 0.04823282753835012  |
    | 0.04986195229399486  |
    | 0.03035462951090517  |
    | 0.012088854133053259 |
    | 0.029288869202029586 |
    | 0.020950992659050018 |
    | 0.021469147566490373 |
    | 0.021381300630990033 |
    | 0.03375819556973361  |
    | 0.020086342302457434 |
    | 0.01682915320908738  |
    | 0.04050573595399061  |
    | 0.015431132136201635 |
    | 0.012754475198474368 |
    | 0.01996127602610097  |
    | 0.023033059794584827 |
    | 0.033993673283573896 |
    | 0.05808789071659969  |
    | 0.06447029324442458  |
    | 0.05786508072405472  |
    | 0.050414988855697916 |
    +----------------------+
    [21 rows x 5 columns]
    


**Quiz Question**: Consider the coefficients of **simple_model**. There should be 21 of them, an intercept term + one for each word in **significant_words**. How many of the 20 coefficients (corresponding to the 20 **significant_words** and *excluding the intercept term*) are positive for the `simple_model`?


```python
positive_significant_words = simple_model.coefficients[simple_model.coefficients['value'] > 0]['index'][1:11]
positive_significant_words
```




    dtype: str
    Rows: 10
    ['love', 'well', 'loves', 'little', 'easy', 'great', 'able', 'perfect', 'old', 'car']



**Quiz Question**: Are the positive words in the **simple_model** (let us call them `positive_significant_words`) also positive words in the **sentiment_model**?


```python
sentiment_model.coefficients.filter_by(positive_significant_words,'index')
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">class</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">love</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.8405057320615117</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">well</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.40107557492332</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">loves</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.9749823125142675</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">little</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.40993162725717625</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">easy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.7349826255674956</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">great</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.7789532883805116</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">able</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.10752802191424699</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">perfect</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0447994204048723</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">old</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.07967490900987591</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">car</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.11965787650766223</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
    </tr>
</table>
[10 rows x 5 columns]<br/>
</div>



# Comparing models

We will now compare the accuracy of the **sentiment_model** and the **simple_model** using the `get_classification_accuracy` method you implemented above.

First, compute the classification accuracy of the **sentiment_model** on the **train_data**:


```python
get_classification_accuracy(sentiment_model, train_data, train_data['sentiment'])
```




    0.976494573364514



Now, compute the classification accuracy of the **simple_model** on the **train_data**:


```python
get_classification_accuracy(simple_model, train_data, train_data['sentiment'])
```




    0.8668150746537147



**Quiz Question**: Which model (**sentiment_model** or **simple_model**) has higher accuracy on the TRAINING set?

Now, we will repeat this exercise on the **test_data**. Start by computing the classification accuracy of the **sentiment_model** on the **test_data**:


```python
get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
```




    0.9221862251019919



Next, we will compute the classification accuracy of the **simple_model** on the **test_data**:


```python
get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
```




    0.8693004559635229



**Quiz Question**: Which model (**sentiment_model** or **simple_model**) has higher accuracy on the TEST set?

## Baseline: Majority class prediction

It is quite common to use the **majority class classifier** as the a baseline (or reference) model for comparison with your classifier model. The majority classifier model predicts the majority class for all data points. At the very least, you should healthily beat the majority class classifier, otherwise, the model is (usually) pointless.

What is the majority class in the **train_data**?


```python
num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print(num_positive)
print(num_negative)
```

    112164
    21252


Now compute the accuracy of the majority class classifier on **test_data**.

**Quiz Question**: Enter the accuracy of the majority class classifier model on the **test_data**. Round your answer to two decimal places (e.g. 0.76).


```python
majority_class_probability = num_positive / len(train_data)
majority_class_probability
```




    0.8407087605684476




```python
num_positive_test  = (test_data['sentiment'] == +1).sum()
num_negative_test = (test_data['sentiment'] == -1).sum()
print(num_positive_test)
print(num_negative_test)
```

    28095
    5241



```python
majority_class_percent = num_positive_test / len(test_data)
majority_class_percent
```




    0.8427825773938085




```python
majority_class_classifier_accuracy = majority_class_probability / majority_class_percent
majority_class_classifier_accuracy
```




    0.997539321669684



**Quiz Question**: Is the **sentiment_model** definitely better than the majority class classifier (the baseline)?
