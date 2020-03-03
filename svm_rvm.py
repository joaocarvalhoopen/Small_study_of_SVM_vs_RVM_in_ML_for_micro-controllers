###############################################################################
#                                 svm_rvm.py
#
# Code that will generate two models, SVM and RVM optimizing the gamma parameter.
###############################################################################
# Author: Joao Nuno Carvalho
# Date:   2020.03.02
# DataSet made available by Simone from the Eloquent Arduino Blog and Github.
# Description: This is a small study. I was curious to know if the optimization
#              of gamma hyper-parameter of the RVM, could make the model from
#              RVM give better results. In particular with this dataset that
#              Simone use to produce the "Audio Word Detection in Arduino" in
#              the table on the following blog post. [Even smaller Machine
#              learning models for your MCU: up to -82% code size]
#  (https://eloquentarduino.github.io/2020/02/even-smaller-machine-learning-models-for-your-mcu/)
#       
#              In this table you can see that RVM compares very well with SVM
#              in terms of accuracy but it is as a smaller memory footprint and
#              a faster inference. This is particularly useful in Machine
#              Learning for micro-controllers.
#              Simone has made a really remarkable work in showing with projects,
#              while documenting on blog posts, that in fact one can do inference
#              of non trivial models of Machine Learning in small micro-controllers.
#              Please see the [Eloquent Arduino](https://eloquentarduino) blog.   
#              The dataset was made available by Simone [eloquentarduino 
#              voice_fft_dataset.py]
#  (https://gist.github.com/eloquentarduino/225039696c59475deef7ea182a7e1569) 
#              In the references I collected links to interesting theoretical
#              and practical information about RVM's. 
#
#              I do some shuffling to the data each time the program is executed
#              but in reality only one time shuffling would be sufficient.
#              This is necessary because I use ranges to separate the train
#              and test dataset.  
#
#              Currently this work is halted, it's half done because because
#              I couldn't install the project sklearn_bayes it gave compilation
#              errors. 
#              https://github.com/AmazaspShumik/sklearn_bayes/
#
#              in Windows 10 or in Ubuntu 19.10 Linux both runing Python Anaconda.
#              The sklearn_bayes package has the fast implementation for RSM
#              that is used in Eloquent Arduino microMLgen at the date that I
#              am writing this comment. 
#              When I locked in the sklearn_bayes project issues there are others
#              with the some problem.
#
# Note: The dataset is from the FFT 32 from Arduino audio word classification
#       and it came from Eloquent Arduino https://eloquentarduino.
#       Project URL: https://eloquentarduino.github.io/2019/12/word-classification-using-arduino/
#       DataSet --> eloquentarduino/voice_fft_dataset.py - https://gist.github.com/eloquentarduino/225039696c59475deef7ea182a7e1569
#       Project of RVM - Relevant Support Machine - Even smaller Machine learning models for your MCU: up to -82% code size
#       https://eloquentarduino.github.io/2020/02/even-smaller-machine-learning-models-for-your-mcu/
#
# SVM - Support Vector Machine 
# RVM - Relevant Vector Machine
#
# The advantage of RVM's is that it uses less memory (less support
# vectors) and the inference execution is faster.
# This is perfect for microcontrollers that have little resources.
#
# Afterwords I tried to use the slower RVM algorithm implementation (2001 paper),
# [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm).
# To install you have to do:
#    pip install https://github.com/JamesRitchie/scikit-rvm/archive/master.zip
#
#

import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from io import StringIO   # StringIO behaves like a file object.
from skrvm import RVC

dataset = '''
// 1

-10.00,-11.00,-14.31,-8.93,-15.84,-34.12,-45.63,-46.56,-66.47,-77.33,-89.85,-98.59,-107.57,-115.89,-120.41,-123.71,-123.71,-121.38,-115.89,-110.24,-100.23,-90.59,-79.95,-68.72,-57.38,-46.01,-36.19,-27.44,-19.98,-14.31,-10.91,-9.76
-10.00,-10.91,-14.19,-19.82,-27.22,-35.01,-44.49,-55.50,13.52,-79.29,-92.08,-101.87,-110.24,-2.83,-121.38,-29.93,-119.72,-123.34,21.67,-106.68,-102.70,-92.08,-81.26,-69.85,-58.79,-47.15,-36.49,-27.44,-20.14,-14.31,-6.35,-6.32
-9.44,-11.09,-14.54,-16.89,-26.99,-30.85,-30.04,-56.91,-22.53,-79.95,-18.56,-101.87,-90.68,-87.62,-115.51,-124.70,-122.71,30.35,-0.94,-110.24,-102.70,-92.82,-81.91,-69.85,-58.32,-47.15,-36.79,-27.66,-20.14,-14.43,-11.00,-9.84
-9.52,-11.18,-14.07,-19.82,-26.77,-34.41,-23.96,-30.57,-68.72,-82.57,-69.80,-98.59,-100.46,-8.48,-119.43,-119.72,-116.72,22.51,37.69,38.23,-8.22,-72.77,-78.64,-68.16,-54.56,-29.66,0.89,1.12,-12.02,-14.19,-10.91,-9.76
-10.00,-11.27,5.51,-20.31,-27.88,-37.08,-46.01,-59.26,-70.98,7.21,-92.08,-101.05,-112.02,32.98,-122.36,-124.70,-123.71,-122.36,-116.83,-109.35,-98.59,-88.37,-78.64,-67.03,-55.97,-44.87,-34.71,-26.10,-18.84,-13.72,-10.73,-9.60
-9.76,-11.00,-14.31,-20.47,-27.88,-36.79,-47.53,-58.79,-70.98,19.66,-87.62,-101.05,-111.13,-117.77,-122.36,-122.71,-107.75,-88.10,-107.41,-108.46,-101.87,-92.82,-81.91,-69.29,-47.50,-19.39,-17.50,-18.07,-17.06,-12.78,-9.30,-6.16
-9.44,-10.82,-9.15,-9.42,-26.10,-4.75,-46.01,-25.87,-70.41,-26.21,-33.42,-98.59,-111.13,-114.95,-65.59,9.98,-92.78,-118.45,-114.01,-107.57,-96.95,-81.68,-70.77,-64.22,-50.80,-46.39,-36.79,-27.66,-20.14,-14.43,-10.82,-9.52
-10.00,-11.09,-14.43,5.85,-27.88,-37.08,-47.91,-59.26,20.84,-81.26,-92.82,-102.70,-111.13,-118.72,-118.45,-123.71,-125.70,-122.36,-117.77,36.45,-54.22,-92.08,-79.95,-67.03,-50.33,-45.63,-35.60,-27.22,-20.14,-14.54,-11.09,-9.92
-8.88,-11.18,-14.66,-20.47,9.82,-37.08,-47.53,-58.79,-68.72,-81.91,-93.57,-103.52,-73.79,-116.83,-121.38,-122.71,-124.70,-120.41,-11.31,-111.13,-101.87,-91.34,-81.26,-64.78,-55.50,-34.22,9.79,6.25,-9.58,-14.07,-10.46,-9.52
-10.00,-11.27,-14.54,-20.31,-27.88,-24.62,-46.77,-58.79,-70.41,20.31,-92.82,-101.05,-109.35,-114.01,-57.76,-89.79,-121.71,-119.43,-112.12,-105.79,-99.41,-89.85,-71.43,-52.95,-54.09,-46.77,-37.08,-27.44,-15.60,-13.96,-11.09,-9.84
-10.00,-11.18,5.04,-20.31,-28.11,-37.38,14.83,-58.79,-70.41,-81.91,34.16,-99.41,-110.24,-117.77,-116.49,35.92,-122.71,-122.36,-116.83,-109.35,-100.23,-89.85,-79.29,-68.16,-56.91,-46.01,-35.90,-26.77,-19.33,-13.96,-10.55,-9.68
-9.52,-11.00,-14.54,-20.31,-17.40,-36.79,-46.77,-56.91,-55.77,-81.26,-92.82,-103.52,-107.57,-117.77,-122.36,-124.70,46.89,-44.05,-116.83,-111.13,-103.52,-53.47,-79.95,-68.72,-57.85,-47.91,-37.08,-12.49,2.92,-11.96,-10.73,-9.60
-8.40,-11.00,-14.19,-2.11,-27.22,-36.79,-47.53,1.88,-69.29,-66.19,-92.82,-82.16,-104.91,-114.01,-122.36,-8.98,-38.91,-117.47,-115.89,-111.13,-98.59,20.05,27.52,-47.32,-57.38,-47.15,-36.49,-27.44,-19.98,-14.43,-11.09,-9.84
-10.08,-9.66,-14.66,-20.31,-28.11,-9.79,-47.15,-58.32,-70.98,-81.26,-90.59,-102.70,-108.46,35.80,-122.36,-124.70,-122.71,-118.45,-117.77,-111.13,-101.05,-89.11,-77.33,-59.15,-53.62,-46.01,-36.49,-27.22,-19.66,-14.19,-10.91,-9.76
-5.76,-11.09,-14.66,-20.31,-27.88,-37.08,-47.53,-4.70,-70.41,-82.57,-0.74,-98.59,-111.13,-117.77,-122.36,-18.96,-124.70,-123.34,-117.77,-110.24,-102.70,-91.34,-81.26,-69.85,-57.38,-46.77,-36.19,-26.99,-19.49,-14.07,-10.64,-9.52
0.00,-11.09,-14.66,-20.31,-27.66,-36.79,-47.53,-58.79,-70.98,-58.32,-92.08,-102.70,-109.35,-118.72,-120.41,-123.71,-83.80,-120.41,-116.83,-107.57,-78.87,-42.33,-60.29,-68.16,-57.85,-46.39,-35.30,-23.65,-16.08,-12.90,-10.46,-9.60

// 6
-9.68,-11.09,-14.43,-19.98,-26.77,-35.90,-46.39,-57.38,-67.60,-78.64,-86.88,-96.12,-103.13,-114.01,-121.38,-122.71,-114.73,-117.47,-113.06,-107.57,-100.23,-91.34,-80.60,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-9.60,-10.91,-14.54,-19.98,-26.55,-35.60,-46.77,-57.85,-69.85,-80.60,-91.34,-101.05,-108.46,-114.95,-119.43,-122.71,-121.71,-119.43,-114.95,-108.46,-100.23,-90.59,-79.95,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-8.64,-10.55,-14.19,-20.14,-27.44,-35.90,-44.87,-57.38,-68.72,-79.95,-91.34,-101.05,-108.46,-114.95,-119.43,-121.71,-121.71,-119.43,-114.95,-108.46,-100.23,-90.59,-79.95,-68.72,-57.38,-46.77,-36.49,-27.22,-19.82,-14.31,-10.91,-9.76
-9.04,-11.00,-13.96,-19.17,-26.77,-36.19,-47.15,-57.85,-66.47,-78.64,-90.59,-101.05,-109.35,-115.89,-120.41,-121.71,-121.71,-120.41,-114.95,-108.46,-100.23,-90.59,-79.95,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-9.44,-10.73,-14.43,-20.14,-27.44,-36.19,-46.01,-57.38,-68.16,-79.95,-90.59,-100.23,-108.46,-114.95,-119.43,-121.71,-121.71,-119.43,-115.89,-109.35,-100.23,-90.59,-79.95,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-8.64,-10.82,-14.43,-20.14,-26.99,-34.12,-46.39,-58.32,-68.72,-78.64,-89.85,-101.05,-109.35,-115.89,-120.41,-122.71,-121.71,-119.43,-114.95,-108.46,-100.23,-91.34,-80.60,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-8.96,-5.72,-12.08,-19.33,-22.31,-29.67,-45.25,-57.38,-69.85,-80.60,-90.59,-98.59,-106.68,-114.01,-119.43,-121.71,-121.71,-119.43,-114.95,-109.35,-100.23,-90.59,-79.95,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-9.44,-10.82,-14.19,-19.82,-27.66,-36.49,-46.01,-55.97,-68.16,-80.60,-91.34,-100.23,-108.46,-114.95,-119.43,-121.71,-122.71,-119.43,-114.95,-108.46,-100.23,-90.59,-79.95,-68.72,-57.38,-46.77,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-7.68,-8.94,-14.07,-20.14,-26.99,-34.12,-45.63,-57.38,-69.85,-81.26,-91.34,-100.23,-108.46,-114.95,-119.43,-122.71,-121.71,-119.43,-114.95,-108.46,-100.23,-90.59,-79.95,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-9.60,-11.00,-14.31,-19.66,-27.66,-36.49,-47.91,-57.38,-64.78,-78.64,-90.59,-102.70,-108.46,-109.30,-116.49,-122.71,-123.71,-121.38,-115.89,-109.35,-101.05,-91.34,-79.95,-68.72,-57.38,-46.77,-36.19,-27.22,-19.82,-14.31,-11.00,-9.84
-8.80,-0.89,-13.25,-19.98,-27.44,-31.15,-45.25,-57.38,-70.41,-80.60,-89.11,-98.59,-107.57,-114.01,-119.43,-121.71,-121.71,-119.43,-114.95,-108.46,-100.23,-90.59,-79.95,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-8.88,-1.61,-10.56,-19.66,-27.66,-28.78,-44.87,-57.38,-69.85,-81.91,-90.59,-98.59,-108.46,-114.01,-119.43,-121.71,-122.71,-119.43,-114.95,-108.46,-100.23,-90.59,-79.95,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.43,-10.91,-9.76
-7.20,-10.55,-14.19,-20.31,-27.22,-35.60,-46.01,-57.85,-69.85,-80.60,-91.34,-101.05,-109.35,-114.95,-119.43,-121.71,-122.71,-120.41,-114.95,-108.46,-100.23,-90.59,-79.95,-69.29,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-10.00,-11.18,-14.43,-19.98,-19.18,-23.44,1.90,-38.57,-69.29,-80.60,-82.43,-99.41,-110.24,-117.77,-119.43,-121.71,-118.72,-118.45,-114.95,-108.46,-100.23,-90.59,-79.95,-68.72,-57.38,-46.39,-36.49,-27.22,-19.82,-14.31,-10.91,-9.76
-9.84,-10.73,-14.78,-20.31,-17.40,-29.07,-46.39,-58.32,-69.29,-77.33,-89.85,-101.87,-111.13,-115.89,-120.41,-122.71,-121.71,-120.41,-114.95,-108.46,-100.23,-90.59,-79.95,-68.72,-57.38,-46.39,-36.19,-27.22,-19.82,-14.31,-10.91,-9.76
-8.56,-11.00,-14.43,-20.14,-26.99,-35.90,-45.63,-57.38,-69.85,-79.95,-89.85,-101.87,-12.45,20.73,-49.92,-121.71,-122.71,-43.07,-113.06,-111.13,-99.41,-73.52,-79.95,-69.29,-57.85,-47.15,-36.49,-27.44,-19.98,-14.31,-10.91,-9.76

// 7
-9.92,2.95,-13.49,-19.82,-25.21,-37.08,-47.15,-57.38,-65.34,-77.33,-90.59,-101.87,-109.35,-115.89,-121.38,-123.71,-122.71,-118.45,-111.18,-104.91,-96.95,-88.37,-76.67,-67.03,-56.44,-46.77,-36.49,-26.10,-19.17,-14.54,-10.46,-9.92
-8.96,-10.10,-14.54,-19.82,-26.32,-37.08,-47.15,-57.85,-68.16,-78.64,-89.85,-98.59,-107.57,-113.06,-116.49,-120.71,-122.71,-121.38,-116.83,-110.24,-101.87,-91.34,-80.60,-68.72,-56.44,-46.77,-36.49,-26.77,-19.98,-14.31,-11.00,-9.84
-1.52,-10.91,-14.43,-19.82,-20.97,-35.30,-47.53,-58.32,-70.41,-81.26,-91.34,-101.05,-110.24,-116.83,-121.38,-123.71,-123.71,-121.38,-117.77,-110.24,-101.05,-89.11,-78.64,-68.16,-58.32,-47.15,-35.90,-27.22,-19.49,-14.43,-10.91,-9.84
-9.04,-11.09,-13.60,-18.03,-27.22,-37.08,-47.15,-54.56,-28.73,-72.08,-67.57,-101.87,-111.13,-117.77,-122.36,-124.70,-124.70,-121.38,-115.89,-109.35,-97.77,-89.11,-79.95,-68.72,-52.21,-46.01,-36.79,-26.55,-20.14,-14.43,-10.91,-9.60
-9.76,-10.82,-14.54,-19.66,-26.77,-29.96,-46.77,-58.32,-69.85,-80.60,-89.11,-101.87,-105.79,20.73,-122.36,-122.71,-124.70,-101.81,-116.83,-111.13,-101.87,-89.11,-79.29,-67.60,-55.97,-43.35,-34.41,-25.88,-18.68,-13.60,-10.46,-9.44
-7.04,-7.33,-14.19,-20.31,-27.44,-36.49,-42.21,-53.15,-34.92,-55.70,-84.65,-91.19,-89.79,-78.20,-85.16,-94.78,-111.74,-108.66,-116.83,-110.24,-101.87,-92.08,-79.95,-67.60,-57.38,-46.01,-35.60,-27.66,-19.49,-14.54,-11.00,-9.68
-8.88,-10.91,-14.31,-19.66,-26.10,-31.45,-38.02,-51.74,-68.16,-81.91,-92.82,-102.70,-111.13,-117.77,-121.38,-123.71,-124.70,-121.38,-117.77,-110.24,-101.87,-92.08,-81.26,-68.72,-56.91,-45.63,-35.90,-27.22,-19.49,-13.96,-11.00,-9.44
-9.60,-8.94,-14.43,-19.98,-27.44,-36.79,-46.77,-56.44,-69.29,-81.26,-91.34,-82.98,-109.35,-114.95,-117.47,-122.71,-121.71,-117.47,-73.49,-87.12,-85.44,-77.97,-71.43,-61.96,-53.15,-43.35,-33.82,-25.21,-19.17,-14.43,-11.00,-9.92
-6.32,-10.82,-13.96,-20.31,-27.66,-36.79,-47.53,-56.91,-68.72,-79.29,-92.08,-27.11,-111.13,-116.83,-117.47,-117.72,-123.71,-119.43,-117.77,-88.01,-73.94,-91.34,-79.95,-70.41,-55.03,-47.15,-36.79,-18.07,-7.31,-10.91,-7.24,-6.48
-9.92,-10.73,-14.54,-20.14,-27.88,-36.79,-46.77,-56.44,-67.60,-81.26,-88.37,-101.05,-110.24,-117.77,-121.38,-123.71,-122.71,-118.45,-116.83,-109.35,-101.87,-92.82,-78.64,-69.85,-58.79,-47.53,-36.49,-22.75,-17.06,-13.02,-5.36,-9.36
-10.00,-11.18,-14.66,-18.52,-24.09,-27.89,-43.35,-54.56,-64.22,-64.22,-88.37,-100.23,-110.24,-117.77,-120.41,-121.71,-120.71,-120.41,-109.30,-110.24,-97.77,-92.08,-79.95,-68.72,-57.85,-45.63,-36.49,-27.22,-19.82,-14.31,-11.00,-9.76
-9.60,-8.67,-14.31,-19.82,-27.22,-36.49,-47.15,-58.32,-70.41,-81.91,-73.52,-81.34,-65.79,-108.35,-121.38,-124.70,-124.70,-122.36,-117.77,-110.24,-98.59,-89.85,-77.33,-69.85,-58.32,-45.25,-36.19,-25.65,-20.14,-13.96,-11.09,-9.76
-9.60,-11.00,-14.19,-19.82,-27.22,-35.90,-46.39,-57.38,-69.29,-79.95,-92.08,-100.23,-109.35,-114.01,-117.47,-113.73,-121.71,-120.41,-115.89,-109.35,-99.41,-90.59,-76.01,-68.16,-58.32,-46.39,-35.90,8.25,-16.24,-14.31,-11.18,-9.60
-9.76,-11.00,-14.07,-18.03,-26.10,-32.93,-41.07,-33.86,-67.03,-77.33,-91.34,-101.87,-110.24,-116.83,-121.38,-120.71,-119.72,-113.55,-113.06,-110.24,-101.87,-91.34,-79.29,-68.72,-56.44,-46.77,-36.49,-27.22,-19.98,-14.31,-10.91,-9.76
-8.96,-10.55,-14.43,-19.98,-26.55,-35.01,-47.15,-56.91,-68.72,-78.64,-91.34,-98.59,-109.35,-115.89,-118.45,-121.71,-122.71,-120.41,-114.95,-108.46,-101.05,-89.85,-79.29,-69.29,-57.38,-46.39,-36.79,-27.22,-18.36,-14.43,-11.00,-9.92
-9.28,-10.55,-14.19,-19.82,-27.66,-37.08,-47.53,-58.79,-70.41,-76.01,-84.65,-93.66,-102.24,-90.45,-113.55,-120.71,-123.71,-121.38,-116.83,-108.46,-96.12,-62.38,-77.33,-67.60,-58.32,-47.15,-36.19,-27.22,-19.82,-13.49,-11.09,-9.60
-10.00,-11.09,-14.19,-18.52,-26.77,-27.59,-45.63,-57.85,-69.85,-81.26,-92.08,-99.41,-92.46,-112.12,-96.91,-123.71,-123.71,-119.43,-109.30,-110.24,-96.12,-92.08,-78.64,-69.85,-58.32,-46.39,-35.90,-27.44,-19.98,-14.19,-10.82,-9.92
-10.00,-11.18,-14.66,-20.31,-27.88,-37.08,-47.15,-54.09,-41.12,-64.87,-86.14,-95.30,-104.02,-109.30,-113.55,-98.77,-122.71,-121.38,-116.83,-110.24,-101.05,-88.37,-79.29,-67.03,-55.03,-45.63,-36.19,-27.66,-19.49,-14.31,-10.82,-9.92
-9.44,-11.00,-14.43,-19.82,-26.77,-32.63,-13.69,-56.44,-69.29,-81.26,-89.11,-100.23,-109.35,-116.83,-122.36,-109.74,-109.74,-116.49,-66.90,-53.34,-73.94,-86.14,-72.08,-68.72,-58.32,-47.15,-36.19,-26.55,-19.98,-13.49,-11.00,-8.80
-8.88,-8.94,-14.19,-19.66,-26.55,-35.01,-44.11,-53.15,-62.53,-74.70,-86.88,-96.12,-105.79,-113.06,-117.47,-119.72,-120.71,-115.51,-112.12,-105.79,-101.87,-91.34,-79.95,-67.60,-58.32,-47.15,-36.49,-26.99,-19.82,-14.31,-10.82,-9.84
'''

def printDataSetTestVsPred(model, X_test):
    y_pred = model.predict(X_test)
    print("y_test:  ", y_test)
    print("y_pred:  ", y_pred)

def genBestSVM(X_train, y_train, X_test, y_test):
    topModel    = None
    topAccTrain = 0
    topAccTest  = 0
    topGamma    = 0
    topNumSupportVecPerClass = 0
 
    # Because the dataset is small, model generation is fast, so we can
    # generate and search in 1000 models with different gamma's.
    for i in range(1, 1000):  # 1000
        # clf = SVC(kernel='linear').fit(X_train, y_train)
        # clf = SVC(kernel='linear', gamma=0.001).fit(X_train, y_train)
        gammaVal = 0.000001 * i
        clf = SVC(kernel='rbf', gamma=gammaVal).fit(X_train, y_train) 
    
        numSupportVectors = clf.n_support_  # Per Class

        y_pred = clf.predict(X_train)
        accTrain = accuracy_score(y_train, y_pred)

        y_pred = clf.predict(X_test)
        accTest = accuracy_score(y_test, y_pred)
        
        print("SVM: gamma_val: {0:.6f}    acc_X_train: {1:.3f}   acc_X_test: {2:.3f}   num_support_vectors: {3}".format(
            gammaVal, accTrain, accTest, numSupportVectors))

        # We want a small difference between the train accuracy and the
        # test accuracy, so that neither one is overfitting,
        # while maximizing the absolute train value.
        deltaCurr = abs(accTrain - accTest)
        if (topModel == None) or (deltaCurr < 0.1 and topAccTrain < accTrain): 
            topAccTrain = accTrain
            topAccTest  = accTest
            topModel = clf
            topGamma = gammaVal
            topNumSupportVecPerClass = numSupportVectors  # Per class.

    return (topModel, topAccTrain, topAccTest, topGamma, topNumSupportVecPerClass)

def genBestRVM(X_train, y_train, X_test, y_test):
    topModel    = None
    topAccTrain = 0
    topAccTest  = 0
    topAlpha    = 0
    topNumSupportVecPerClass = 0
 
    alphaVal = 0.00000001
    # Because the dataset is small, model generation is fast, so we can
    # generate and search in 12 models with different alpha's.
    for i in range(1, 13):
        alphaVal *= 5
        # alphaVal = 0.000001
        clf = RVC(kernel='rbf', alpha=alphaVal).fit(X_train, y_train) 

        # Relevance Vectors
        numSupportVectors = [] # Per class

        y_pred = clf.predict(X_train)
        accTrain = accuracy_score(y_train, y_pred)

        y_pred = clf.predict(X_test)
        accTest = accuracy_score(y_test, y_pred)
        
        print("RVM: alpha_val: {0:.8f}    acc_X_train: {1:.3f}   acc_X_test: {2:.3f}   num_support_vectors: {3}".format(
            alphaVal, accTrain, accTest, numSupportVectors))

        printDataSetTestVsPred(clf, X_test)

        # We want a small difference between the train accuracy and the
        # test accuracy, so that neither one is overfitting,
        # while maximizing the absolute train value.
        deltaCurr = abs(accTrain - accTest)
        if (topModel == None) or (deltaCurr < 0.1 and topAccTrain < accTrain): 
            topAccTrain = accTrain
            topAccTest  = accTest
            topModel = clf
            topAlpha = alphaVal
            topNumSupportVecPerClass = numSupportVectors  # Per class.

    return (topModel, topAccTrain, topAccTest, topAlpha, topNumSupportVecPerClass)


if __name__ == '__main__':
    
    strFinal = ""
    for line in dataset.splitlines():
        if len(line) == 0:
            pass
        elif len(line) == 4:
            y_curr = int(line[3])
        elif len(line) > 10:
          strFinal += str(y_curr) + "," + line + "\n"

    c = StringIO(strFinal)
    data = np.loadtxt(c, delimiter=',', usecols=range(33))

    # Total number of features: 32
    # Total number of cases in all classes: 16 + 16 + 20

    X = data[:, 1:]  # select columns 1 through end
    y = data[:, 0]   # select column 0, the class

    data, target = shuffle(X, y, random_state = 0) # 0 # 2   
    
    X_train, X_test = data[:-10, :], data[-10:, :] # 10
    y_train, y_test = target[:-10], target[-10:]

    #######
    # Generate the best SVM optimizing the Gamma hyper-parameter.
    
    topModel, topAccTrain, topAccTest, topGamma, topNumSupportVecPerClass = genBestSVM(X_train, y_train, X_test, y_test)
    print("\nSVM: top_gamma: {0:.6f}    acc_X_train: {1:.3f}   acc_X_test: {2:.3f}   num_support_vectors: {3}".format(
        topGamma, topAccTrain, topAccTest, topNumSupportVecPerClass))
    
    # Here just to compare to see if the 3 classes were present in the target test dataset.
    y_pred = topModel.predict(X_test)
    print("y_test:  ", y_test)
    print("y_pred:  ", y_pred)

    #######
    # Generate the best RVM optimizing the alpha hyper-parameter.

    topModel, topAccTrain, topAccTest, topAlpha, topNumSupportVecPerClass = genBestRVM(X_train, y_train, X_test, y_test)
    print("\nRVM: top_alpha: {0:.6f}    acc_X_train: {1:.3f}   acc_X_test: {2:.3f}   num_support_vectors: {3}".format(
        topAlpha, topAccTrain, topAccTest, topNumSupportVecPerClass))
    
    # Here just to compare to see if the 3 classes were present in the target test dataset.
    # y_pred = topModel.predict(X_test)
    # print("y_test:  ", y_test)
    # print("y_pred:  ", y_pred)

    printDataSetTestVsPred(topModel, X_test)

    print("...end")

