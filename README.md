# XCM
The cross customer model (XCM) is a machine learning modelling 
system designed to take features from openrtb auctions and determine the probability of a good behaviour auction from some predetermined definition of good behaviour.

XCM is currently supported for both python2 and python3.

## Features

XCM is designed to be extendable, there are 4 extendable classes defined in `xcm.core.base_classes`:

#### XCMReader
The `XCMReader` class is designed to read from a source of data on a per-day basis.
The reader classes should implement an interface returning data that can be parsed by an `xcm.core.records.XCMRecord` and an `xcm.core.records.AdDataRecord`.
The reader classes are also responsible for labelling the data they provide and supplying a realistic labeling method.
New Reader classes should be added for new data types or data sources.
 
No reader classes are provided by default

#### XCMClassifier
The `XCMClassifier` class is designed to provide an interface to the machine learning techniques. 
It is responsible for arithmatic manipulation of the model by use of the `partial_fit` method.
It evaluates the result of the model for some given input though the `predict` method
The classifier also requires a `forget` method to reduce the false positive affects of overfitting a model.


A `BOPRClassifier` is provided. (See class for more info) 

#### XCM
The `XCM` base class acts as the base class for a model. 
A model has a name and versions and works with the classifier to return information about the model.
This class also works with the `XCMReader` classes to complete training exercises and the `XCMStore` classes to persist all relevant data to the databases. 


Two representations are provided:
1. The `XCMModel` is a high(er) performance model that does not train and can be considered immutable
2. The `XCMTrainingModel` is a trainable model that can update it's classifier

The `XCMClassifier`s methods are dependent on the implementations of the `XCM.load` and `XCM.serialise` methods.

#### XCMStore
The `XCMStore` class is designed to store in perpetuity the XCM model data and any learned techniques. 
The XCM models are continuous and so all historic learnings compound.
The XCMStore classes have to be able to implement CRUD features. 
The implementations and methods of the `XCMStore` classes are impacted by the methods of the `XCM` classes.

An `S3XCMStore` is provided to store XCM models as files on S3. 
It is possible to store each `XCM` classes models in different stores (prodution vs development).
Two helper methods have been provided to create each of the different stores (production vs dev):
```python
>>> from xcm.stores.utils import get_active_model_store, get_training_model_store
>>> training_store = get_training_model_store()
>>> active_store = get_active_model_store()
```


XCM is a cross customer model, meaning that the data from multiple customers is combined to produce a model. 
In this way, features can become either prominent irrespective of customer or are customer dependent.
 
## Training

XCM training is done in 3 stages:

1. train the baseline model. The baseline model is the continuous learning component and is continually forgotten to reduce overfitting
2. train the current model. The active model takes the last week of data and does no forgetting. It adds bias to recent data.
3. create an active model. Transforms an `XCMTrainingModel` into an `XCMModel` for better performance for successive evaluations.

A single `xcm.core.training.build_xcm_model` has been provided to encompass this work:
```python
>>> from xcm.core.training import build_xcm_model
>>> from xcm.stores.utils import get_active_model_store, get_training_model_store
>>> from xcm.core.base_classes import XCMReader
>>> build_xcm_model(get_training_model_store(), get_active_model_store(), XCMReader())
```
The XCM training needn't occur more frequently than once per day.


## Prediction
To predict with an XCM model, call the predict method on an XCM class:
```python
>>> from xcm.stores.utils import get_active_model_store
>>> xcm_store = get_active_model_store()
>>> newest_model_id = xcm_store.list()[0]
>>> model = xcm_store.retrieve(newest_model_id)
>>> data = {}  # include stuff here from an XCMRecord
>>> model.predict(data)
```

##### Deprecated

The XCM package comes with functionality to create a ROC curve and AUC charts