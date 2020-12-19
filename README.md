# Probabilistic Future Prediction for Video Scene Understanding
This EECV 2020 paper proposes a novel deep learning method for autonomous driving. This method controls a car and predicts the future only from video data. What does future prediction have to do with autonomous driving? Well, a lot.
 
Being able to predict possible scenarios does indeed help while driving, right? Predicting the future is one of the greatest capabilities of humans. While driving it helps you decide when to slow down, accelerate, or break. At an intersection, you know that another car may come from the left, or someone may cross the street. The car and the pedestrian could also interact with each other (*Multi-Agent interaction*). What will happen is not completely certain: there is no **one future** but there are **many** possible **futures**. This is why the authors handle this problem not as deterministic, but as probabilistic.

You may ask "has this not be done already? Why is this paper different?". You see, the problem of future prediction is by far not as popular as image classification, but of course it has already been addressed.  But previous works had limitations.  
 * No end-to-end methods. This means that they did not carry out the whole process from the beginning - from the sequence of 2D data -  to the end - to the prediction and the car controls.  
 * Fail to model Multi-Agent interactions, by assuming for dynamics agents such as vehicles and pedestrians to act independently from each other.  
 * Work with low-resolution input or simulated and unrealistic data. As you can imagine, guessing what an image depicts it's not so easy if the image looks like the one on the left, instead of the one on the right. <p float="right"> <img src="images/apolloscape_lr.png" width="450" /> <img src="images/apolloscape_hr.png" width="450" /> </p> On top of that, using video frames from simulated or unrealistic data does simplify the task. So, even if the evaluation is still indicative, it is controversial whether the model is capable of facing real-life situations. Reality is indeed much more challenging, unpredictable, diverse, and complicated than a game or a simulation.

### Input & Output
First, let’s reason about the method in terms of input and output. The input is a sequence of images, of 2D frames from videos. The predicted driving controls are velocity, acceleration, steering angle, and angular velocity. Then, how to show the predicted future? With a video, a sequence of  2D images as the input?  It may work. But we can do it smarter: let's predict semantic segmentation, depth, and optical flow. In other words,
 <p> <img align="right" src="images/segmentation_depth_and_flow.gif" alt> </p>
 
* where is what. Where in the image do we have a car, a pedestrian, traffic light, sign, road, lane... This is *semantic segmentation* (upper right).
* How far from us is each object, hereby estimating its *depth* (lower left).
* from where to where each object did move with respect to the previous frame.  In other words,  estimate the movement or flow of certain particles in the image. This is called *optical flow* (lower right). 

You can read this article without problems after this short introduction. But if you are not familiar with these three concepts, you can inform yourself  - there are many great blogs out there.

### Network
Let’s take a look at the data flow and see how the network is constructed. To allow a good grasp of the gradient flow, the red squares in the images show the variables involved in the loss computation.

<img src="images/Network_loss_circles.png" />

The Perception module takes as input 5 frames of the past 1 second and encodes them into segmentation, flow, and depth information. That information is concatenated i.e. fused together for each input frame to produce the perception feature <img src="https://render.githubusercontent.com/render/math?math=x_{t-i}">.  
Those features are fed to the Dynamics module, a 3D convolutional network that contains a novel module, the Temporal Block. This module extracts in parallel local and global features. On a local level, it separates the convolutions acquiring in parallel spatial features, vertical motion, horizontal motion, and overall motion. I will not dive deeper into it, since its analysis in the paper is pretty straightforward. The Dynamics module outputs the spatio-temporal representation <img src="https://render.githubusercontent.com/render/math?math=z_{t}">, which contains information up to present time <img src="https://render.githubusercontent.com/render/math?math=t">.  
This representation goes into the Prediction module. Here a generator, a convolutional GRU, outputs future codes <img src="https://render.githubusercontent.com/render/math?math=g_{t%2Bi}"> for each future time step <img src="https://render.githubusercontent.com/render/math?math=t%2Bi">. The codes are then decoded to predict segmentation, depth, and flow maps for each one of the 10 future time steps, covering a total of 2 seconds.  
The spatio-temporal representation <img src="https://render.githubusercontent.com/render/math?math=z_{t}">  goes also into the Control module. This module predicts the car controls -  velocity, acceleration, steering angle, and angular velocity - for each future time step.  
At this point, the whole method does operate in a deterministic setting (the input to the generator for each time step is the zero vector).  How to make the method probabilistic? By adding the module containing the future and present distribution, the probabilistic module. More about it in the probabilistic loss section.

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AR%28g%29+%26%3D+%5Cfrac%7B1%7D%7Bn%7D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cell%28y_i%2Cg%28x_i%29%29%5C%5C%0A%26%3D%5Cfrac%7B1%7D%7B2n%7D+%28%5Cmathbf%7BX%7D%5Cboldsymbol%7Bw%7D-%5Cmathbf%7By%7D%29%5ET+%28%5Cmathbf%7BX%7D%5Cboldsymbol%7Bw%7D-%5Cmathbf%7By%7D%29%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
R(g) &= \frac{1}{n} \sum_{i=1}^{n} \ell(y_i,g(x_i))\\
&=\frac{1}{2n} (\mathbf{X}\boldsymbol{w}-\mathbf{y})^T (\mathbf{X}\boldsymbol{w}-\mathbf{y})
\end{align*}
">

![](https://render.githubusercontent.com/render/math?math=e^{i %2B\pi} =x%2B1)

<img src="https://render.githubusercontent.com/render/math?math=e^{i %2B\pi} =x%2B1">

![\begin{align*}
R(g) &= \frac{1}{n} \sum_{i=1}^{n} \ell(y_i,g(x_i))\\
&=\frac{1}{2n} (\mathbf{X}\boldsymbol{w}-\mathbf{y})^T (\mathbf{X}\boldsymbol{w}-\mathbf{y})
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AR%28g%29+%26%3D+%5Cfrac%7B1%7D%7Bn%7D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cell%28y_i%2Cg%28x_i%29%29%5C%5C%0A%26%3D%5Cfrac%7B1%7D%7B2n%7D+%28%5Cmathbf%7BX%7D%5Cboldsymbol%7Bw%7D-%5Cmathbf%7By%7D%29%5ET+%28%5Cmathbf%7BX%7D%5Cboldsymbol%7Bw%7D-%5Cmathbf%7By%7D%29%0A%5Cend%7Balign%2A%7D%0A)
