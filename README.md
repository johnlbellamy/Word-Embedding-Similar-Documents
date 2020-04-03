pnnl-demo
==============================

Sentiment and semantic tagging to enable knowledge management using Solr.

REQUIREMENTS
------------
* Assumes you have the Anaconda distribution of Python installed. It also assumes you have Anaconda installed in:

/home/${USER}/anaconda3. You can change this in the MakeFile (Line 11) and in the environment.yml (Line 9). 

* This workflow is designed for a Linux environment. 

You can build this workflow with make. The steps to reproduce what is demoed are below:

USAGE
------------

### Downloads raw data and tensorflow USE module
1) make data

### Creates Anaconda environment and downloads pip reqs. 
2) make environment

### Performs the cleaning and feature engineering
3) make features 

### Creates the embedded sentiment model and tagged data
4) make use

### Finishes the process by assigning sentiment to model. The processed data from this step is uploaded to Mongo
5) make use_sentiment 

### OPTIONAL: SVD
1) follow steps 1-3 above then:
2) make svd
3) make svd_sentiment


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    │
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.sh
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
