Panka
====
Predicting antibiotic resistance 

Antimicrobial resistance is one of the most pressing global health concerns, and machine learning algorithms can be a powerful tool in the fight against it by learning about antimicrobial resistance mechanisms from DNA sequence data without any prior knowledge. The first and most fundamental step in building a machine learning model is extracting the features from the sequencing data. Various feature extraction methods have been proposed for this task, however, combining different feature extraction methods for the classification task in sequencing data remains a computational challenge. By integrating various features extracted from the pan-genome, we present a method for detecting antimicrobial resistance using sequencing data. In addition, our method can identify both established and potential resistance determinants from the corresponding signatures, enabling researchers to extract meaningful scientific insights. We apply the proposed method as well as other existing feature extraction methods to Escherichia coli and Klebsiella pneumoniae bacteria species. The results indicate that our model is more accurate than conventional approaches as well as the state-of-the-art classification method for sequencing data. 

----------
Installing
----------

Requirements:

* Python 3.6 or greater
* numpy
* scipy
* numba
* panta (https://github.com/amromics/panta)
* more packages in python notebook

---------------
How to use Panka
---------------
``` r


For more realistic examples and Python scripts to reproduce the results
in our paper are available at the producibility directory.

-------
License
-------

The Panka package is 3-clause BSD licensed.

This code was tested on 
Python 3.6, 3.7; numpy version 1.19.2; scipy version 1.5.3; numba version 0.52.0 

