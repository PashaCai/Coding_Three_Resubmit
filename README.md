# Coding_Three_Resubmit
Code &amp; Read_Me for resubmit

<p>URL for Video:https://youtu.be/U_ykVmANZJU?si=nqC25z1Vl4-_O5_n</p>
<p>URL for The CIFAR-10 dataset:https://www.cs.toronto.edu/~kriz/cifar.html</p>

<h2>Purpose:</h2>
<p>To achieve automatic classification of images after they are provided, or to predict categories, by means of a trained model.</p>
<h2>Motivation:</h2>
<p>My aim is to produce automatic classification of images through machine learning, because my undergraduate major is graphic design, so I am interested in this aspect of graphic classification, which will help me when I go to produce some similar graphics or capture images.</p>

<h2>Process:</h2>
<p>In my study I focused on tensoflow's tutorial on classifying cifar graphics using CNN (Convolutional Neural Network), due to technical issues I used the simpler FNN for machine learning and used the same cifar-10 dataset as in the tutorial as my training set. In the process I learned how to use matplotlib for data visualisation, which was very interesting, matplotlib as a plotting library for python is very nice to use, and it allows for many types of graphical objects, even simple 3D animations, which is quite different from the adobe indesign graphs that I used to create.

I visualised the images of the dataset using matplotlib and visualised their grey scale channels separately, (0-1) pixel distribution of the images and tabulated the distribution and regions of the grey scale values of the images.

The next part of the study was about neural networks, which is honestly a bit boring and difficult for someone like me with no basic knowledge, but I referred to the tutorial "Fully Connected Neural Networks Explained" by Shining (2022) and Zhingjun(2022)'s tutorial "The Most Detailed Tutorial on Building Fully Connected Neural Network Models on the Site", these tutorials teach you to understand and use the FNN in a much more understandable way. That's why I ended up using FNN.

While the tensorflow tutorial uses CNN and the Keras Sequential API and is probably more accessible, I couldn't use it because he didn't teach CNN in a very detailed way, and it would be plagiarism to just copy and paste it.

The learning process behind this feels a lot like going back to high school to me, such as the step of converting images to vectors for training NN models. My understanding of the code here in layman's terms is to keep the image in the first dimension and multiply the subsequent dimensions, conceivably pulling them into a vector, because the image is two-dimensional, e.g. 32*32.

This conversion is much like converting [[1,1][1,1]] to [1,1,1,1,1,1,1]], which can be more simply understood as a formal conversion to convert a two-dimensional image to one-dimensional data, making it easier for neural networks to perform machine learning.

At the same time, according to what I have learnt, I think I can understand this simple neural network I use as y=k1(kx+b)+b1 in this form, a kind of simple binary equation is more conducive to my learning. You can say that the training process is to find the parameters of k, b, k1, b1, and after finding these parameters, you can imagine that you can find a y (label) by giving an x (data), that is to say, it can classify a picture automatically when given a picture.

My understanding of sigmoid is that the final output is actually the probability, so to speak, of each category. For example, I output a 2*1 vector [0.1, 0.5], where 0.1 represents the probability of the first category, and 0.5 represents the probability of the second category, that is to say, the image belongs to whichever category has the highest probability.

At the same time, I also learned about the loss function (loss), in fact, the model is through the loss to optimise, it can be commonly understood as follows: through the function above we can get a y by inputting an x, we get a y and the real y to do comparisons (cross-entropy loss function (Cross-Entropy Loss)). The difference between them is the loss. The process of constantly training the model is actually optimising the loss so that the difference between them gets smaller and smaller so that the predictions are more accurate.

In the end my test using FCNN was only 28.94% accurate and I think I was missing a part which was using the trained model to analyse the images and classify them, which was a bit different from what I had envisaged.

After talking to my instructor I realised that I was lacking in this assignment and that I wasn't using some of the key points from the class, so I returned to re-learn and re-read the parts of the class that had sections on CNNs and datasets and focused on the tutorial from tensorflow (2022) "Using How to Train a Simple Convolutional Neural Network (CNN) to classify CIFAR images", while an article on CNNs by Sorokina (2017) and a detailed explanation of Convolutional Neural Networks by Shining (2022) also helped me a lot.

In the subsequent work I improved the last part of the code, I used Convolutional Neural Network (CNN) to classify CIFAR images, and added predict to test the trained model, and finally compared the two different types of data, I concluded that the validation accuracy of CNN is much higher than that of my previous FNN for both methods, based on the randomly visualisation of the prediction results of the selected test set, it can be seen that the accuracy of its CNN prediction is higher.</p>

<h2>Evaluation and Reflection：</h2>
<p>After delving into and comparing the performance of Fully Connected Neural Networks (FNNs) and Convolutional Neural Networks (CNNs) on an image classification task, I noticed a significant accuracy difference: the CNNs tested at 70.72% accuracy, whereas the FNNs tested at only 28.94% accuracy. This huge gap prompted me to explore the reasons behind it. After referring to Dufeng's (2022) article on CNN infrastructure and JM's (2023) analysis of the reasons for CNN's superior performance on images, I came up with some key insights.

First, the fundamental difference between convolutional and fully connected layers is the way they process data. While fully-connected layers process information by means of dense connections, convolutional layers employ a more fine-grained mechanism of local connections and weight sharing. This difference is not only reflected in the architectural design, but also closely related to the characteristics of the image itself. Key features such as edges and corners usually occupy only a small portion of an image, and while the direct relationship between distant pixels in an image is weak, there is a strong correlation between local pixels. It is due to this local connectivity and weight sharing mechanism that CNNs are able to capture these local features more efficiently with a smaller number of parameters, thus demonstrating greater capability in image recognition tasks.

Secondly, CNN's ability to achieve excellent results in image classification is also due to the design of its multi-layer convolutional operation.CNN is able to extract image features from simple to complex layer by layer by stacking multiple convolutional layers and performing pooling operations of different sizes. This layer-by-layer learning process from low-level features such as edges to high-level features such as object shapes and textures is difficult to achieve with fully-connected layers due to the large number of parameters. It is this ability to capture multi-level features of an image that makes CNNs not only efficient but also more accurate in classifying when processing image data.

By combining the results of my project experiments with the analyses of Dufeng (2022) as well as JM (2023), I have gained a deeper understanding of why CNNs outperform FNNs in image classification. This not only gave me a deeper insight into the differences in neural network architectures, but also provided me with better options for future project choices in image processing and machine learning.

At the end I would like to point out one thing, my thinking has not really been very correct, I have a very stereotypical approach to code and this has led to a very monotonous output, although I was able to visualise some of the dataset's content with the help of matplotlib, it was not aesthetically pleasing overall. I felt very strange at length while writing this code, unlike painting, which gives me a sense of freedom, a sense of easy writing, but not code. Although I think they are similar in nature, both of them are created from scratch, but the code gives me a sense of constraint, I think this is caused by my lack of personal ability, the use of code to write installations and images should also be a free thing, but my ability to limit it, I need to make up for this in the future in the future learning of the shortcomings.</p>


<h2>Reference List</h2>

<p>1.Du, F (2022) ‘[Deep Learning] Basics - CNN: Image Classification’ , CSDN blog, 10 April. Available at: https://blog.csdn.net/fengdu78/article/details/124089531 (Accessed: 5 October 2023).</p>

<p>2.JM-0808 (2023) ‘Reasons why CNNs perform well on images’ , CSDN blog, 31 May. 
Available at: https://blog.csdn.net/weixin_56033928/article/details/130973233 (Accessed: 5 October 2023).</p>

<p>3.Shining, L (2022) ‘Fully Connected Neural Networks Explained’ , CSDN blog, 25 September. Available at: https://blog.csdn.net/ShiningLeeJ/article/details/126676581 (Accessed: 11 May 2023).</p>

<p>4.Sorokina, K (2017) Image Classification with Convolutional Neural Networks. Available at:https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8 (Accessed: 3 October 2023)</p>

<p>5.Shining, L (2022) ‘Convolutional Neural Networks Explained’ , CSDN blog, 13 September. Available at: https://blog.csdn.net/ShiningLeeJ/article/details/126827739 (Accessed: 4 October 2023).</p>

<p>6.TensorFlow (2022) Convolutional Neural Network, CNN. Available at: https://www.tensorflow.org/tutorials/images/cnn?hl=zh-cn (Accessed: 3 October 2023).</p>

<p>7.Zhingjun, T(2022) ‘The Most Detailed Tutorial on Building Fully Connected Neural Network Models on the Site’. CSDN blog, 03 May. Available at: http://t.csdnimg.cn/BwpeO (Accessed: 12 May 2023).</p>

