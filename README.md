# Small study of SVM vs RVM in ML for micro-controllers
Code that will generate two models SVM and RVM, optimizing the gamma hyper-parameter.

## Description
This is a small study. I was curious to know if the optimization of gamma hyper-parameter of the RVM, could make the model from RVM give better results. In particular with this dataset that Simone use to produce the "Audio Word Detection in Arduino" in the table on the following blog post. [Even smaller Machine learning models for your MCU: up to -82% code size](https://eloquentarduino.github.io/2020/02/even-smaller-machine-learning-models-for-your-mcu/) <br>
In this table you can see that RVM compares very well with SVM in terms of accuracy but it is as a smaller memory footprint and a faster inference. This is particularly useful in Machine Learning for micro-controllers.<br>
Simone has made a really remarkable work in showing with projects, while documenting on blog posts, that in fact one can do inference of non trivial models of Machine Learning in small micro-controllers. Please see the [Eloquent Arduino](https://eloquentarduino.github.io/) blog.   
The dataset was made available by Simone [eloquentarduino voice_fft_dataset.py](https://gist.github.com/eloquentarduino/225039696c59475deef7ea182a7e1569) . <br>
In the references I collected links to interesting theoretical and practical information about RVM's.

## Points to note
Because the data is small 52 cases and high dimensional (32 features) the optimization of gamma is heavily dependent on the shuffle lotter. If you omit the random_state = 0 in the shuffle call you can see for yourself that you can generate results for the best model, from 60% to 90% accuracy in the test dataset. But you should try to find one that has a similar value between the accuracy of the train and test dataset's, small delta (no overfitting), you should maximize the absolute train accuracy and you should look to see if the 3 classes are represented in the test dataset.      

## Screen shoot
![SVM - model generation and gamma optimization](./svm_output.png)

## Status of this small study
Currently this work is halted, it's half done because because I couldn't install the project [sklearn_bayes](https://github.com/AmazaspShumik/sklearn_bayes/) it encountered compilation errors, in Windows 10 or in Ubuntu 19.10 Linux both running Python Anaconda.<br>
The sklearn_bayes package has the fast implementation for RVM that is used in Eloquent Arduino microMLgen at the time that I am writing this words. When I looked in the sklearn_bayes project issues there are others with the some problem. So maybe is a question of time until the problem is resolved and I can resume with this small study.

## Dependencies
[Project sklearn_bayes for the fast implementation of RVM](https://github.com/AmazaspShumik/sklearn_bayes/) <br>
[Project micromlgen from Eloquent Arduino](https://github.com/eloquentarduino/micromlgen) <br>

# Glossary
* SVM - Support Vector Machine 
* RVM - Relevant Vector Machine

## References
* [Eloquent Arduino](https://eloquentarduino.github.io/)
* [Project from were I learned about RVM - Relevant Support Machine - Even smaller Machine learning models for your MCU: up to -82% code size](https://eloquentarduino.github.io/2020/02/even-smaller-machine-learning-models-for-your-mcu/)
* [DataSet is from Simone from project Word classification using Arduino and MicroML](https://eloquentarduino.github.io/2019/12/word-classification-using-arduino/)
* [The dataset is from the FFT 32 Arduino audio word classification](https://gist.github.com/eloquentarduino/225039696c59475deef7ea182a7e1569)
* [Project sklearn_bayes for the fast implementation of RVM](https://github.com/AmazaspShumik/sklearn_bayes/)
* [Wikipedia - Relevance vector machine](https://en.wikipedia.org/wiki/Relevance_vector_machine)
* [Tipping's webpage on Sparse Bayesian Models and the RVM](http://www.miketipping.com/sparsebayes.htm)
* [Paper Sparse Bayesian Learning and the Relevance Vector Machine by Michael E. Tipping 2001](http://jmlr.csail.mit.edu/papers/v1/tipping01a.html)
* [Paper Sequential Sparse Bayesian Learning) was discovered later by Faul and Tipping 2003. This documents the fast training algorithm for RVM](http://www.miketipping.com/papers/met-fastsbl.pdf)
* [JamesRitchie - scikit-rvm - Other implementation of RVM with scikit-learn interface but with slower algorithms](https://github.com/JamesRitchie/scikit-rvm)
* [RVM are explained in Section 7.2 of Christopher Bishops's Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fcmbishop%2Fprml%2F)  

## License
My code is MIT Open Source license, the dataset was made available by Simone from Eloquent Arduino, see references.<br>
This work is just a small study that I wish to continue because I see real potential in RVM - Relevant Support Machines in the context of small micro-controllers.

## Have fun!
Best regards, <br>
Joao Nuno Carvalho <br>
