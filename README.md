# Coding_Three_Resubmit
Code &amp; Read_Me for resubmit

<h2>Purpose:<h2/>
To achieve automatic classification of images after they are provided, or to predict categories, by means of a trained model.

Motivation:
My aim is to produce automatic classification of images through machine learning, because my undergraduate major is graphic design, so I am interested in this aspect of graphic classification, which will help me when I go to produce some similar graphics or capture images.

Process:
In my study I focused on tensoflow's tutorial on classifying cifar graphics using CNN (Convolutional Neural Network), due to technical issues I used the simpler FNN for machine learning and used the same cifar-10 dataset as in the tutorial as my training set. In the process I learned how to use matplotlib for data visualisation, which was very interesting, matplotlib as a plotting library for python is very nice to use, and it allows for many types of graphical objects, even simple 3D animations, which is quite different from the adobe indesign graphs that I used to create.

I visualised the images of the dataset using matplotlib and visualised their grey scale channels separately, (0-1) pixel distribution of the images and tabulated the distribution and regions of the grey scale values of the images.

The next part of the study was about neural networks, which is honestly a bit boring and difficult for someone like me with no basic knowledge, but I referred to the tutorial "Fully Connected Neural Networks Explained" by Shining (2022), which teaches you to understand and use the FNN in a much more understandable way. That's why I ended up using FNN.

While the tensorflow tutorial uses CNN and the Keras Sequential API and is probably more accessible, I couldn't use it because he didn't teach CNN in a very detailed way, and it would be plagiarism to just copy and paste it.

The learning process behind this feels a lot like going back to high school to me, such as the step of converting images to vectors for training NN models. My understanding of the code here in layman's terms is to keep the image in the first dimension and multiply the subsequent dimensions, conceivably pulling them into a vector, because the image is two-dimensional, e.g. 32*32.

This conversion is much like converting [[1,1][1,1]] to [1,1,1,1,1,1,1]], which can be more simply understood as a formal conversion to convert a two-dimensional image to one-dimensional data, making it easier for neural networks to perform machine learning.

At the same time, according to what I have learnt, I think I can understand this simple neural network I use as y=k1(kx+b)+b1 in this form, a kind of simple binary equation is more conducive to my learning. You can say that the training process is to find the parameters of k, b, k1, b1, and after finding these parameters, you can imagine that you can find a y (label) by giving an x (data), that is to say, it can classify a picture automatically when given a picture.
My understanding of sigmoid is that the final output is actually the probability, so to speak, of each category. For example, I output a 2*1 vector [0.1, 0.5], where 0.1 represents the probability of the first category, and 0.5 represents the probability of the second category, that is to say, the image belongs to whichever category has the highest probability.

At the same time, I also learned about the loss function (loss), in fact, the model is through the loss to optimise, it can be commonly understood as follows: through the function above we can get a y by inputting an x, we get a y and the real y to do comparisons (cross-entropy loss function (Cross-Entropy Loss)). The difference between them is the loss. The process of constantly training the model is actually optimising the loss so that the difference between them gets smaller and smaller so that the predictions are more accurate.

In the end my test using FCNN was only 28.94% accurate and I think I was missing a part which was using the trained model to analyse the images and classify them, which was a bit different from what I had envisaged.

After talking to my instructor I realised that I was lacking in this assignment and that I wasn't using some of the key points from the class, so I returned to re-learn and re-read the parts of the class that had sections on CNNs and datasets and focused on the tutorial from tensorflow (2022) "Using How to Train a Simple Convolutional Neural Network (CNN) to classify CIFAR images", while an article on CNNs by Sorokina (2017) and a detailed explanation of Convolutional Neural Networks by Shining (2022) also helped me a lot.

In the subsequent work I improved the last part of the code, I used Convolutional Neural Network (CNN) to classify CIFAR images, and added predict to test the trained model, and finally compared the two different types of data, I concluded that the validation accuracy of CNN is much higher than that of my previous FNN for both methods, based on the randomly visualisation of the prediction results of the selected test set, it can be seen that the accuracy of its CNN prediction is higher.


Evaluation and Reflection：
For this difference in accuracy (CNN test_acc 70.72%, FNN test_acc 28.94%), after referring to Dufeng's (2022) article on CNN infrastructure and JM's (2023) analysis of the reasons why CNNs perform so well on images, I've come to the conclusion that the biggest problem lies in the difference between convolutional and fully connected layers. fully-connected layer difference.

Local connectivity and weight sharing: whereas fully connected layers are densely connected, convolutional layers only use convolutional kernels to process locally, which actually corresponds to the characteristics of the image. In visual recognition, critical image features, edges, corners, etc. occupy only a small portion of the whole image, and the likelihood that there is a connection or influence between pixels that are far apart is very low, while local pixels have a strong correlation. This mechanism both reduces the number of parameters of the model and improves the ability of the convolutional neural network to perceive multidimensional data such as images.

Multi-Layer Convolutional Operation: In convolutional neural networks, high-level features of an image can be continuously extracted by stacking multiple convolutional layers with pooling operations of different sizes. Through multi-layer convolutional operations, the neural network can learn layer by layer from simple features (e.g., edges) to complex abstract features (e.g., shape of the object, texture, etc.). (Fully connected layers are not possible due to the large number of parameters).

Therefore, CNN applications generally require local correlation of the convolved objects, which is exactly what images have, so it makes sense that CNNs are used in the image field.

Second I would like to point out one thing, my thinking has not really been very correct, I have a very stereotypical approach to code and this has led to a very monotonous output, although I was able to visualise some of the dataset's content with the help of matplotlib, it was not aesthetically pleasing overall. I felt very strange at length while writing this code, unlike painting, which gives me a sense of freedom, a sense of easy writing, but not code. Although I think they are similar in nature, both of them are created from scratch, but the code gives me a sense of constraint, I think this is caused by my lack of personal ability, the use of code to write installations and images should also be a free thing, but my ability to limit it, I need to make up for this in the future in the future learning of the shortcomings.


Reference List
1.Shining, L (2022) ‘Fully Connected Neural Networks Explained’ , CSDN blog, 25 September. Available at: https://blog.csdn.net/ShiningLeeJ/article/details/126676581 (Accessed: 11 May 2023).

2.TensorFlow (2022) Convolutional Neural Network, CNN. Available at: https://www.tensorflow.org/tutorials/images/cnn?hl=zh-cn (Accessed: 3 October 2023).

3.Sorokina, K (2017) Image Classification with Convolutional Neural Networks. Available at:https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8 (Accessed: 3 October 2023)

4.Shining, L (2022) ‘Convolutional Neural Networks Explained’ , CSDN blog, 13 September. Available at: https://blog.csdn.net/ShiningLeeJ/article/details/126827739 (Accessed: 4 October 2023).

5.Du, F (2022) ‘[Deep Learning] Basics - CNN: Image Classification’ , CSDN blog, 10 April. Available at: https://blog.csdn.net/fengdu78/article/details/124089531 (Accessed: 5 October 2023).

6.JM-0808 (2023) ‘Reasons why CNNs perform well on images’ , CSDN blog, 31 May. 
Available at: https://blog.csdn.net/weixin_56033928/article/details/130973233 (Accessed: 5 October 2023).
