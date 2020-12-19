# Probabilistic Future Prediction for Video Scene Understanding
This EECV 2020 paper proposes a novel deep learning method for autonomous driving. This method controls a car and predicts the future only from video data. What does future prediction have to do with autonomous driving? Well, a lot.
 
Being able to predict possible scenarios does indeed help while driving, right? Predicting the future is one of the greatest capabilities of humans. While driving it helps you decide when to slow down, accelerate, or break. At an intersection, you know that another car may come from the left, or someone may cross the street. The car and the pedestrian could also interact with each other (*Multi-Agent interaction*). What will happen is not completely certain: there is no **one future** but there are **many** possible **futures**. This is why the authors handle this problem not as deterministic, but as probabilistic.

You may ask "has this not be done already? Why is this paper different?". You see, the problem of future prediction is by far not as popular as image classification, but of course it has already been addressed.  But previous works had limitations.  
 * They were not end-to-end methods. This means that they did not carry out the whole process from the beginning - from the sequence of 2D data -  to the end - to the prediction and the car controls.  
 * They also failed to model Multi-Agent interactions. Because they assumed dynamics agents such as vehicles and pedestrians to act independently from each other.  
 * And, they worked with low-resolution input or simulated and unrealistic data. As you can imagine, guessing what an image depicts it's not so easy if the image looks like the one on the left, instead of the one on the right. <p float="right"> <img src="images/apolloscape_hr.png" width="450" /> <img src="images/apolloscape_lr.png" width="450" /> </p>
On top of that, using video frames from simulated or unrealistic data does simplify the task. So, even if the evaluation is still indicative, it is controversial whether the model is capable of facing real-life situations. Reality is indeed much more challenging, unpredictable, diverse, and complicated than a game or a simulation.

### Input & Output
First, let’s reason about the method in terms of input and output. The input is a sequence of images, of 2D frames from videos. The predicted driving controls are velocity, acceleration, steering angle, and angular velocity. Then, how to show the predicted future? With a video, a sequence of  2D images as the input?  It may work. But we can do it smarter: let's predict semantic segmentation, depth, and optical flow. In other words,
* where is what. Where in the image do we have a car, a pedestrian, traffic light, sign, road, lane... This is *semantic segmentation*.
* How far from us is each object, hereby estimating its *depth*.
* from where to where each object did move with respect to the previous frame.  In other words,  estimate the movement or flow of certain particles in the image. This is called *optical flow*. 
 <p> <img src="images/segmentation_depth_and_flow" alt>
    <em>image_caption</em>
</p>
You can read this article without problems after this short introduction. But if you are not familiar with these three concepts, you can inform yourself  - there are many great blogs out there.




<img src="images/apolloscape_hr.png" alt="drawing" width="500"/>   <img src="images/apolloscape_lr.png" alt="drawing" width="500"/> 
