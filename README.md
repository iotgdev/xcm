# XCM
The cross campaign model (XCM) is a machine learning modelling 
system designed to take features from openrtb auctions and determine the probability of a good behaviour auction from some predetermined definition of good behaviour.

XCM is currently supported for both python2 and python3.

## Features

### XCMModel 
A model has a name and works with the classifier to return information about the model.
This class uses the `XCMStore` classes to persist all relevant data to the databases. 

### XCMStore
The `XCMStore` class is designed to store in perpetuity the XCM model data and any learned techniques. 
The XCM models are continuous and so all historic learnings compound.

XCM is a cross campaign model, meaning that the data from multiple campaigns is combined to produce a model. 
In this way, features can become either prominent irrespective of campaign or are campaign dependent.

## Prediction
To predict with an XCM model, call the predict method on an XCM class:
```python
>>> from xcm.models.store import XCMStore
>>> xcm_store = XCMStore()
>>> model = xcm_store.list(as_list=True)[0]
>>> data = {}  # include stuff here from an XCMRecord
>>> model.predict(data)
```
