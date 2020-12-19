# Probabilistic Future Prediction for Video Scene Understanding
This EECV 2020 paper proposes a novel deep learning method for autonomous driving. This method controls a car and predicts the future only from video data. What does future prediction have to do with autonomous driving? Well, a lot.
 
Being able to predict possible scenarios does indeed help while driving, right? Predicting the future is one of the greatest capabilities of humans. While driving it helps you decide when to slow down, accelerate, or break. At an intersection, you know that another car may come from the left, or someone may cross the street. The car and the pedestrian could also interact with each other (*Multi-Agent interaction*). What will happen is not completely certain: there is no **one future** but there are **many** possible **futures**. This is why the authors handle this problem not as deterministic, but as probabilistic.

You may ask "has this not be done already? Why is this paper different?". You see, the problem of future prediction is by far not as popular as image classification, but of course it has already been addressed.  But previous works had limitations.  
 * No end-to-end methods. This means that they did not carry out the whole process from the beginning - from the sequence of 2D data -  to the end - to the prediction and the car controls.  
 * Fail to model Multi-Agent interactions, by assuming for dynamics agents such as vehicles and pedestrians to act independently from each other.  
 * Work with low-resolution input or simulated and unrealistic data. As you can imagine, guessing what an image depicts it's not so easy if the image looks like the one on the left, instead of the one on the right. <p float="right"> <img src="images/apolloscape_lr.png" width="450" /> <img src="images/apolloscape_hr.png" width="450" /> </p> On top of that, using video frames from simulated or unrealistic data does simplify the task. So, even if the evaluation is still indicative, it is controversial whether the model is capable of facing real-life situations. Reality is indeed much more challenging, unpredictable, diverse, and complicated than a game or a simulation.

## Input & Output
First, let’s reason about the method in terms of input and output. The input is a sequence of images, of 2D frames from videos. The predicted driving controls are velocity, acceleration, steering angle, and angular velocity. Then, how to show the predicted future? With a video, a sequence of  2D images as the input?  It may work. But we can do it smarter: let's predict semantic segmentation, depth, and optical flow. In other words,
 <p> <img align="right" src="images/segmentation_depth_and_flow.gif" alt> </p>
 
* where is what. Where in the image do we have a car, a pedestrian, traffic light, sign, road, lane... This is *semantic segmentation* (upper right).
* How far from us is each object, hereby estimating its *depth* (lower left).
* from where to where each object did move with respect to the previous frame.  In other words,  estimate the movement or flow of certain particles in the image. This is called *optical flow* (lower right). 

You can read this article without problems after this short introduction. But if you are not familiar with these three concepts, you can inform yourself  - there are many great blogs out there.

## Network
Let’s take a look at the data flow and see how the network is constructed. To allow a good grasp of the gradient flow, the red squares in the images show the variables involved in the loss computation.

<img src="images/Network_loss_circles.png" />

The Perception module takes as input 5 frames of the past 1 second and encodes them into segmentation, flow, and depth information. That information is concatenated i.e. fused together for each input frame to produce the perception feature <img src="https://render.githubusercontent.com/render/math?math=x_{t-i}">.  
Those features are fed to the Dynamics module, a 3D convolutional network that contains a novel module, the Temporal Block. This module extracts in parallel local and global features. On a local level, it separates the convolutions acquiring in parallel spatial features, vertical motion, horizontal motion, and overall motion. I will not dive deeper into it, since its analysis in the paper is pretty straightforward. The Dynamics module outputs the spatio-temporal representation <img src="https://render.githubusercontent.com/render/math?math=z_{t}">, which contains information up to present time <img src="https://render.githubusercontent.com/render/math?math=t">.  

This representation goes into the Prediction module. Here a generator, a convolutional GRU, takes as initial hidden state <img src="https://render.githubusercontent.com/render/math?math=z_{t}"> and outputs future codes <img src="https://render.githubusercontent.com/render/math?math=g_{t%2Bi}"> for each future time step <img src="https://render.githubusercontent.com/render/math?math=t%2Bi">. The codes are then decoded to predict segmentation, depth, and flow maps for each one of the 10 future time steps, covering a total of 2 seconds.  
The spatio-temporal representation <img src="https://render.githubusercontent.com/render/math?math=z_{t}">  goes also into the Control module. This module predicts the car controls -  velocity <img src="https://render.githubusercontent.com/render/math?math=\v">, acceleration <img src="https://render.githubusercontent.com/render/math?math=\dot{v}">, steering angle <img src="https://render.githubusercontent.com/render/math?math=\theta">, and angular velocity <img src="https://render.githubusercontent.com/render/math?math=\dot{\theta}">- for each future time step.

At this point, the whole method does operate in a deterministic setting (the input to the generator for each time step is the zero vector).  How to make the method probabilistic? By adding the module containing the future and present distribution, the probabilistic module. More about it in the probabilistic loss section.

## Losses
Here the focus is on the exciting mathematical formulations. Yes, math. Math is powerful. If you are still reading, let’s go through the losses. The total loss is constructed by weighting and adding up three single losses:

 <img align="center" src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AL+%26%3D+%5Clambda_%7Bfp%7D+L_%7Bfuture_pred%7D+%2B%5Clambda_%7Bc%7D+L_%7Bcontrol%7D+%2B+%5Clambda_%7Bp%7D+L_%7Bprobabilistic%7D%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
L &= \lambda_{fp} L_{future\_pred} +\lambda_{c} L_{control} + \lambda_{p} L_{probabilistic}
\end{align*}
">

### The prediction loss. 
First, where does the ground truth come from? From a teacher model. The encoders in the perception module are originally from two well-known autoencoder architectures ([] for segmentation and depth, [] for optical flow). Those autoencoders are pretrained jointly and then separated. While the encoders end up making part of the Perception module, the Decoders are used as a teacher module. The teacher takes the perception features of future frames and decodes them providing a pseudo ground-truth (note that these are other decoders as the ones in the prediction module).  
For each one of the future time steps <img src="https://render.githubusercontent.com/render/math?math=N_f">, the predicted maps are compared to the output of a teacher model. For segmentation, via cross-entropy bootstrapping loss[] <img src="https://render.githubusercontent.com/render/math?math=L_{segm}">,  for depth via scale-invariant depth loss[] <img src="https://render.githubusercontent.com/render/math?math=L_{depth}">,  and for optical flow via the Huber loss <img src="https://render.githubusercontent.com/render/math?math=L_{flow}">. 

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AL_%7Bfuture_pred%7D+%26%3D+%5Clambda_%7Bs%7D+%5Csum_%7Bi%3D0%7D%5E%7BN_f+-1%7D+%5Cgamma%5Ei+L_%7Bsegm%7D%5E%7Bt%2Bi%7D+%2B+%5Clambda_%7Bd%7D+%5Csum_%7Bi%3D0%7D%5E%7BN_f+-1%7D+%5Cgamma%5Ei+L_%7Bdepth%7D%5E%7Bt%2Bi%7D+%2B+%5Clambda_%7Bf%7D+%5Csum_%7Bi%3D0%7D%5E%7BN_f+-1%7D+%5Cgamma%5Ei+L_%7Bflow%7D%5E%7Bt%2Bi%7D+%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
L_{future\_pred} &= \lambda_{s} \sum_{i=0}^{N_f -1} \gamma^i L_{segm}^{t+i} + \lambda_{d} \sum_{i=0}^{N_f -1} \gamma^i L_{depth}^{t+i} + \lambda_{f} \sum_{i=0}^{N_f -1} \gamma^i L_{flow}^{t+i} 
\end{align*}
">

As you may have already seen if you are familiar with reinforcement learning, the loss for 3 timesteps in the future and the loss for 10 timesteps in the future do not have the same influence. The losses of each timestep <img src="https://render.githubusercontent.com/render/math?math=t"> are not simply added together; before they are multiplied with a value <img src="https://render.githubusercontent.com/render/math?math=\gamma^t">, where <img src="https://render.githubusercontent.com/render/math?math=\gamma"> is the weighted discount term. Since <img src="https://render.githubusercontent.com/render/math?math=0<\gamma<1">, <img src="https://render.githubusercontent.com/render/math?math=\gamma^t"> gets always smaller for increasing <img src="https://render.githubusercontent.com/render/math?math=0<t<N_f">. In this way, the further we go into the future, the less does a loss count in the overall loss. Indeed, while driving the next half-second is more relevant than the future in 2 seconds, right?

 
 
### The control loss. 
Here the predicted controls are extrapolated and compared to the expert’s control actions <img src="https://render.githubusercontent.com/render/math?math=v_{t}^{gt}"> and <img src="https://render.githubusercontent.com/render/math?math=\theta_{t}^{gt}">.  The authors have access to these actions thanks to the company Wayve, the workplace of some of the paper authors. Exploiting expert's actions as ground truth is called *Conditional Imitation learning*. 

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AL_%7Bcontrol%7D+%26%3D++%5Csum_%7Bi%3D0%7D%5E%7BN_c-1%7D+%5Cgamma_%7Bc%7D%5Ei+%28%28v_%7Bt%2B1%7D%5E%7Bgt%7D-%28v_t+%2B+i+%5Cdot%7Bv%7D_t%29%29%5E2+%2B++%28%5Ctheta_%7Bt%2B1%7D%5E%7Bgt%7D-%28%5Ctheta_t%2B+i+%5Cdot%7B%5Ctheta%7D_t%29%29%5E2%29%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
L_{control} &=  \sum_{i=0}^{N_c-1} \gamma_{c}^i ((v_{t+1}^{gt}-(v_t + i \dot{v}_t))^2 +  (\theta_{t+1}^{gt}-(\theta_t+ i \dot{\theta}_t))^2)
\end{align*}
">

Each future timestep is weighted via the discount term gamma analogous to the prediction loss.

### The probabilistic loss. 
This is the most interesting loss, be ready for the exciting math. This loss is built according to the conditional variational approach. (I aim for a clear, short, and concise idea of conditional variational approaches, without the intention of completeness. It is closely related to Bayesian inference and variational autoencoders, about which there are whole books and articles.). Let’s see how this approach works.
 
Suppose we are interested in a certain probability distribution. For example the probability distribution <img src="https://render.githubusercontent.com/render/math?math=p(Z|x)"> of the future random variable *Z* given certain observations *x*. We do not know the distribution *p* but we have access to a part of the data generated by this distribution (our dataset). In this case, we can approximate *p* with a variational distribution <img src="https://render.githubusercontent.com/render/math?math=q_{\theta}(z)"> of a latent variable *z*, which is parametrized by parameters <img src="https://render.githubusercontent.com/render/math?math=\theta">. We just need to regress the parameters <img src="https://render.githubusercontent.com/render/math?math=\theta"> with a neural network and force the two distributions *p* and *q* to be similar via KL divergence.
 
 <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AKL%28q%5C%7Cp%29%3D+E_q+%5Cleft%5Blog%7B%5Cfrac%7Bq%28Z%29%7D%7Bp%28Z%7Cx%29%7D+%7D+%5Cright%5D%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
KL(q\|p)= E_q \left[log{\frac{q(Z)}{p(Z|x)} } \right]
\end{align*}
">

Easy, right?

On this line of thoughts, the authors define two distributions:
 * **Present distribution** <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}(\mu_{t, present}, \sigma_{t, present}^2)"> parametrized by network *P*, which takes as input the spatio-temporal representation <img src="https://render.githubusercontent.com/render/math?math=z_t">. This distribution predicts the future given observations up to the present time *t*. It represents what COULD happen given the past observations.
 * **Future distribution** <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}(\mu_{t, future}, \sigma_{t, future}^2)"> parametrized by network *F*, which takes as input the spatio-temporal representation <img src="https://render.githubusercontent.com/render/math?math=z_{t+N}">. It is conditioned on <img src="https://render.githubusercontent.com/render/math?math=z_{t+N}">, so it is aware of both past and future information. It represents what DOES actually HAPPEN in that precise observation or video sequence. <p> <img align="right" src="https://upload.wikimedia.org/wikipedia/commons/1/12/Bimodal-bivariate-small.png" width="250" alt> </p>
Both distributions are 8-dimensional, diagonal Gaussian distributions. Here,  8  uncorrelated dimensions i.e. random variables are enough to represent the future in latent space. Most important, both distributions are **multi-modal**. They cover the different modes i.e. possibilities of the future. Remember? There is no “one future” but there are more possible futures with similar likelihood. On the right you can see a 2-dimensional (bivariate) multimodal distribution, where each "bump" represents a different mode.


#### How are the distributions integrated into the method? 

<img src="images/Probabilistic_prediction_modules.png" width="550"/>

A sample from the distribution <img src="https://render.githubusercontent.com/render/math?math=\eta ~ \mathcal{N}(\mu, \sigma^2)">is given as input to the generator (the convolutional GRU) in the Prediction module.  This input is constant for every future time step. At train time it is sampled from the future distribution, while at test time from the present distribution.  
So at train time, the future distribution sees the past and the future of video sequences. Since the data are diverse, the future distribution will naturally be multi-modal. But how can the present distribution be multimodal,  if it never sees in person the modes of the future? By forcing the present distribution to be similar to the future distribution via KL divergence. This happens in the probabilistic loss.

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AL_%7Bprobabilistic%7D+%3D+D_%7BKL%7D+%28F%28%5Ccdotp%7C+Z_t%2C+...%2C+Z_%7Bt%2BN_f%7D%5C%7CP%28%5Ccdotp%7C+Z_t%29%29%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
L_{probabilistic} = D_{KL} (F(\cdotp| Z_t, ..., Z_{t+N_f}\|P(\cdotp| Z_t))
\end{align*}
">
 
As you see, the present distribution is not used at train time, but it is still trained. At test time, we can sample from it and generate diverse and realistic futures.

## Data and Training
The model is trained on 8x 2080Ti NVIDIA GPUs with frames of size 224x480 (256x512 for CItyscapes), which is a relatively high resolution. Inference runs on a single GPU in real-time. This fact has high significance in autonomous driving because it means that the method can be used in real-life applications. 

The whole model besides the teacher network is trained on non-public data from the British company Wayve. Before training, the encoders of the perception model and the teacher model are pretrained as an autoencoder.  
Except for the optical flow autoencoder, which is a PWC-Net [] pretrained off-the-shelf, the autoencoders are trained with data collected from CityScapes [14], Mapillary Vistas [48], ApolloScape [29], and Berkeley Deep Drive [67]. These data were collected during real driving scenarios and present enough realistic situations. For example, they show seasonal (winter, summer), weather (rain), lighting (day, night), and viewpoint changes. Among the covered 6 continents, in the images below you can see China on the left and the USA on the right.
<p float="right"> <img src="images/apolloscape.gif" width="450" /> <img src="images/berkeleyDeepDrive.png" width="450" /> </p>
 
In autonomous driving and in DL in general it is crucial to work with diverse and realistic data. What the network has never seen, it is unlikely to learn. Imagine, you have never seen rain in your life and it suddenly starts pouring down while you are behind the steering wheel. It may become more difficult for you to accomplish the driving task. Still, you could manage. Human’s generalization capabilities are indeed amazing, and transferring them to DL models is an open challenge.


## Results
Since there are no other end-to-end methods to compare against, the authors simply substitute the Dynamics module with other other spatio-temporal architectures. They choose the convolutional GRU in [], 3D ResNet[], and the 3D inception network from []. In this way, they obtain 3 deterministic and 3 probabilistic networks against which they can compare their method.  form the evaluation their reach the following conclusions:
 * Their method achieves the best performance in both the deterministic and the probabilistic case, according to the unified perception metric.  This motivates both the probabilistic module and the temporal block. They are the reasons for the performance improvement.
 
 *  The probabilistic approach improves the performance of every method
 
 * Their method generates the most accurate and diverse futures (see the diversity distance metric DDM).
 
This kind of evaluation is a bit restrictive. So to compare with a broader scientific community they also evaluate their method on future semantic segmentation with the Cityscapes dataset. Cityscapes is not as challenging as the previous data collection because it does not present bad weather or night conditions. It is entirely collected in Germany. Here some considerations from their comparison against Navabi[5] and Chiu[6].
 * This method achieves the best performance.
 * The future is predicted for 5 and 10 steps in the future, covering respectively 0.29s and 0.59s. This is a relatively high value. Navabi[5] and Chiu[6] were presented for 3 or 4 steps.
 * This method achieves a score of 0.464 mean IoU for 5 steps prediction. The state of the art on Cityscapes for present semantic segmentation is around 0.85 mIoU[4], almost double as much. As you can see, the problem of future semantic segmentation has not been solved yet and is more challenging than the present case. 

Let’s now take a look at some qualitative examples taken from the [blog article](https://wayve.ai/blog/predicting-the-future) on the Wayve website.
#### Traffic Line
Here our car waits in a line by a traffic light. Sampling from the present distribution generates two futures.
 ![](https://cdn.sanity.io/images/rmgve84j/production/1e97a7b6ff2bb02042d2c8722f4cc9a65a2c2c98-2500x889.gif?w=1920&h=683&fit=crop)
 The upper one shows a future where the line does not move. The second prediction is more interesting. If the car in front of us starts moving, our car moves along. This example shows that the method is able to grasp the meaning behind this multi-agent interaction, which may seem trivial, but is not.

#### Pedestrian Crossing
This example shows a crosswalk at a red light. Some people cross the street from the left, and at the present time a new pedestrian appears in the image from the right.
![](https://cdn.sanity.io/images/rmgve84j/production/d5695c48c69314f3c62fc0be3954ad80db51e792-1440x224.gif)
 In the future prediction, this person keeps walking in the same direction and crosses the street. While the overall motion is correctly represented, the oddly-shaped red silhouettes that represent pedestrians show room for improvement.

#### Complex Intersection
Here we are at an intersection, with another vehicle and a bus coming from the opposite direction.
![](https://cdn.sanity.io/images/rmgve84j/production/66625d679d021aaf943d78e4b082768da9fc1afc-2195x1026.gif?w=1920&h=897&fit=crop)
The generated futures show our car waiting, slowing down, turning right, or turning left. Take a look at the future segmentation of the right turn. Here the method seems to fail. Did you notice? The bus almost disappears, instead of continuing straight or turning right. Since our car is also turning right, here it is particularly important to understand how the bus will move. As I mentioned before, 2 seconds in the future is pretty far. If it was a real driving scenario, the network may have corrected itself. Seeing what the bus does after 0.5 seconds would have improved the prediction of the future 1.5 seconds.
