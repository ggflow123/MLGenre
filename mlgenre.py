# Yuanzhe Liu
# mlgenre.py
# run machine learning algorithm on the dataset advised by generate_dataset.py
# 1/24/2020
# uaage: python3 mlgenre.py filename percentage seed

import sys
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def WriteConfusionMatrix(fname, result, labels, method):
    Fname = "result" + "_" + str(method) + "_" + str(fname[:-4]) + ".csv"
    file = open(Fname, "w+")
    file.write((",".join(list(map(str, labels)))).replace('"', '') + "\n")
    for i in range(len(labels)):
        result[i].append(labels[i])
        file.write((",".join(list(map(str, result[i])))).replace('"', '') + "\n")
    return

def SupportVectorClassification(train, trainLabel, test):
    classifierSVC = SVC(gamma='auto')
    classifierSVC.fit(train, trainLabel)
    print('fit complete')
    prediction = classifierSVC.predict(test)
    print('prediction')
    return prediction

# Read in the file and generate different sessions
def ReadData(filename, seed, pt):
    data2 = pd.read_csv(filename)
    attributes = list(data2.columns)  # the list of attribute
    data = data2.sample(frac=1, random_state=seed)  # shuffle the data

    # extract the label from the data set
    listlabel = data[data.columns[0]]
    labelList = listlabel.tolist()
    labels = list(dict.fromkeys(labelList)) # remove repeated labels
    newdata = pd.DataFrame()

    # Preprocessing the data using pandas
    for i in range(len(attributes)):
        if i == 0:
            # newdata = pd.concat([newdata, data[attributes[i]]], axis=1)
            continue
        else:
            a = attributes[i]
            # if the data is the string, process one hot encoding
            if isinstance(data[a][0], str):
                one_hot = pd.get_dummies(data[a], prefix=a)
                newdata = pd.concat([newdata, one_hot], axis=1)
            else:
                newdata = pd.concat([newdata, data[a]], axis=1)
    # split the data into train, and test
    datalist = newdata.values.tolist()
    newattributes = list(newdata.columns)
    # shuffle the data
    c = int(pt * len(datalist))
    # set up train, and test
    train = datalist[:c]
    test = datalist[c:]
    trainLabel = labelList[:c]
    testLabel = labelList[c:]
    return train, test, trainLabel, testLabel, newattributes, labels

def main():
    if len(sys.argv) != 4:
        print('Wrong Arguments! Need: file, pt, seed')
    else:
        file = sys.argv[1]
        pt = float(sys.argv[2])
        seed = int(sys.argv[3])
        train, test, trainLabel, testLabel, attributes, labels = ReadData(file, seed, pt)
        result = [[0 for x in range(len(labels))] for y in range(len(labels))]  # confusion matrix
        y_pred = SupportVectorClassification(train, trainLabel, test)

        count = 0
        for i in range(len(testLabel)):
            print(i)
            if y_pred[i] == testLabel[i]:
                count += 1
            result[labels.index(testLabel[i])][labels.index(y_pred[i])] += 1  # adding confusion matrix

        print('accuracy: ', count/len(testLabel))
        print(classification_report(testLabel, y_pred))

        method = 'SVM'
        WriteConfusionMatrix(file, result, labels, method)





if __name__ == '__main__':
    main()