# MLGenre
Machine Learning on Music Genre Classification

Author: Yuanzhe Liu (a student in Oberlin College)

**Grateful to the support by:**

* Professor Adam Eck, Oberlin College
* Professor Eli Stine, Oberlin College



# Description:

This is the project about using machine learning to classify music genres. 

Right now the dataset is the Million Song Subset. This is the link: http://millionsongdataset.com/

The Machine Learning Algorithm is Support Vector Machine. The accuracy is terrible at first, only 3%. Now, it reaches to 23 percent.

Since Every attribute with array doesn't share equal length in each instance, right now I ignore all of them. I will deal with them after.



# Usage:

First, use generate_dataset.py to change the dataset into the form that we can feed into machine learning.

generate_dataset.py: takes in the MillionSongSubset, then return a .csv file with labels and attributes.

```
python3 generate_dataset.py MillionSongSubset
```

Then, we use the simple_data.csv generated by generate_dataset.py to run machine learning:

```
python3 mlgenre.py simple_data.csv 0.75 12345
```

where 0.75 is the percentage of the training dataset, and 12345 is the seed of your choice.

It will generate a confusion matrix in .csv file.



Also, if you would like to manipulate simple_data.csv before run machine learning algorithm on it, feel free to run dataScript.py first:

~~~
python3 dataScript.py simple_data.csv True
~~~

where True at the last specifies whether you would like to categorize the 'tempo' attribute. If True, Yes. If False: No. Then the data will be stored in small_data.csv. Notice that the next step is:

~~~
python3 mlgenre.py small_data.csv 0.75 12345
~~~



# Citation:

```
Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011.
```