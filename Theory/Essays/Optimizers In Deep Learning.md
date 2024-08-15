# An Overview of Main Optimizers in Deep Learning
#DeepLearning #Math #Optimizers 


Article structure:
- Introduction
- Body
	- Batch Gradient Descent
	- Stochastic Gradient Descent
	- Mini-batch Gradient Descent
	- Momentum SGD
	- Nesterov Accelerated Gradient
	- Adagrad
	- RMSprop
	- Adadelta
	- Adam
	- Adamax
	- Nadam
- Conclusion

Enjoy!

---

Parameter tuning based on the Loss Function results is the key part of Deep Learning that gives our models ability to learn. Picking a good optimizer can be a really tricky process full of confusion, so this article will describe all main optimizers in Deep Learning. This article was inspired by the following paper: [An overview of gradient descent optimization algorithms - Sebastian Ruder](https://arxiv.org/pdf/1609.04747), however in this article I try to cover the topic wider, introducing all formulas and their effects.



## Batch Gradient Descent
This optimization algorithm is the first optimization algorithm for the most of readers.
Batch Gradient Descent is the simplest optimizer possible.

Optimization process:
1) Forward Propagation.
2) Cost function on whole dataset.
3) Calculate gradients of cost function w.r.t parameters.
4) Apply gradients to the parameters.

$$\theta = \theta - \eta \nabla J(\theta)$$

However, this optimization algorithm is not efficient for problems with **Large Datasets** that do not fit to the memory and slows down training process in general.
It happens because we apply Mathematical operations (dot product for example) to large matrices. This process may be really exhaustive for our Machine used for model fit.

## Stochastic Gradient Descent (SGD)
Stochastic gradient descent fixes the issues of Gradient Descent, by processing one sample at a time.
It's important to notice that each **epoch** contains $m$ iterations, i.e we perform training on whole data in one **epoch**, but split it on **$m$ *iterations*. However this approach is faster because of simpler computations.

Pseudocode with python-like syntax:

```python
for epoch in range(epochs):
	shuffle(data)
	for sample in data:
		gradients = evaluate_gradients(sample, parameters, loss_fn)
		parameters -= learning_rate * gradients
```

Update rule for SGD:

$$\theta = \theta - \eta \nabla J(\theta , x^{(i)}, y^{(i)})$$

This optimization algorithms is really popular, because of:
1) It's fast
2) It converges minima with *learning rate decay*

This algorithms makes *noisy*[1] but *quick* updates.
Sometimes updates may be too noisy

## Mini-Batch Gradient Descent
This optimization algorithm takes best from both of Batch Gradient Descent and SGD.
It's quick and not noisy.
Key secret here is mini-batch GD perform updates on each *mini-batch*.
It splits our dataset on mini-batches (typically of size from 64 to 256) and processes one mini-batch at iteration. It's faster than Batch Gradient Descent, and less noisy than SGD.
This optimizer works pretty fine because of the reasons above.
$$\theta = \theta - \eta \nabla J(\theta, x^{(i:i+n)}, y^{(i:i+n)})$$
where n - batch size

Pseudocode:

```python
for i in range(epochs):
	shuffle(data)
	for batch in split_batches(data, batch_size=64):
		gradients = evaluate_gradient(loss_fn, batch, params)
		params -= learning_rate * gradients 
```


## Momentum
This is the first optimizer on the list that proposes some interesting concepts and ideas to improve ordinary SGD.
We have problems with all mentioned optimizers:
1) They May do a lot of useless updates
2) They find only **Local Optima**

Momentum proposes intuitive solution.
We could **accumulate gradients** (i.e emulate velocity) and update our parameters with these accumulated gradients.
Thankfully we have a concept of moving average, so we can easily implement it with a formula:

$$v_t = \gamma v_{t-1} + (1 - \gamma)\eta\nabla J(\theta)$$
Where $\gamma$ - momentum term (usually 0.9 or something close to 1)
This $v_t$ stands for velocity at time step $t$.
Note: $t$ stands for iteration. If we have SGD: $T=m$
And we make an update with $v_t$
$$\theta_{t+1} = \theta_t - v_t$$

This method dampens useless oscillations that may occur with really thin and wide bowl-shaped function. (For example it may happen when we do not normalize our input features).
<br>
<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-28_at_3.25.40_PM_Y687HvA.png" width=70%>

Also this method helps resolves problem of Local Optimas as our "ball" that rolls down has some accumulated velocity that can help to jump out of local optima.
<br>
It does it by reducing velocity on axis with a lot of updates in opposite directions.
When Climbing upwards we increase $v_t$ and as a result decrease parameter faster.
However this formula may be counter intuitive, so here's another version of it:

$$v_t = \gamma v_{t-1} - (1-\gamma)\eta \nabla J(\theta_t)$$
$$\theta_{t+1} = \theta_t + v_t$$
those are absolutely the same, but imho this one is much more reasonable.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*2aS_8T7-f5gkoE-3gA9JlA.png" width=50%>

Note: Also you may see the following formula:

$$v_t = \gamma v_{t-1} - \eta \nabla J(\theta _t)$$
Hmmm... But where is the $(1-\gamma)$ term?
With such formulas it's already included to $\eta$
$$\eta = lr(1-\gamma)$$

## Nesterov Accelerated Gradient

Soviet mathematician Yurii Nesterov improved Classical Momentum proposed by Polyak.
Instead of calculating gradient at the current step Nesterov proposed calculating it at the step ahead:

$$v_t = \gamma v_{t-1} - (1-\gamma)\nabla J(\theta_t - \gamma v_{t-1})$$
$$\theta_{t+1} = \theta_t - v_t$$
Although it's just an approximation of "Step Ahead" it works pretty well and does a smarter job, speeding up training process.

Here's an illustration [2] :
<br>
<img src="https://www.ruder.io/content/images/2016/09/nesterov_update_vector.png" widht=30%>
<br>

While CM(classical momentum) makes a small jump in gradient(small blue vector) and makes a big jump in accumulated gradient vector(big blue vector), NAG(Nesterov accelerated gradient) makes a big jump in the direction of accumulated gradients vector (brown vector) and makes a correction (that is to say, adds up gradient at the step ahead).


## AdaGrad

Adagrad differs from Momentum-based optimizers described above, because instead of speeding up or slowing down gradient it changes learning rate ($\eta$) for each parameter.

How does it do it? Let's find out!

Steps of AdaGrad:

1) Accumulate Squared gradients for each parameter:
		$G_{ii} = \sum^T_{t=1}g_{ti}^2$ , where $g_{ti} = \nabla J(\theta_{ti})$
		$G_{ii}$ - diagonal matrix of summed up squared gradients

2) Update parameters with scaled learning rate.
		$\theta_{t+1, i} = \theta_{ti} - {\eta \over \sqrt{G_{tii} + \epsilon}}g_{ti}$
		Note: $\epsilon$ prevents division by zero.
		We update out parameter by dividing our accumulated gradient at diagonal.
		Square root prevents too rapid update.
AdaGrad decreases learning rate on Steep Slopes and Increases it on moderate ones.
That's a really simple idea behind AdaGrad, but despite the simplicity it's efficient.

## RMSprop

RMSprop was proposed by Geoffrey Hinton on his Coursera class and improved AdaGrad optimization algorithms. The main problem of Adagrad is **summation of all gradients through T iterations**. This summation decreased $\eta$ to really small values through $T$ iterations when $T$ was big enough.

RMSprop uses moving average (we've already used it with CM and NAG).

$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma)g^2_t$$
$$\theta_{t+1} = \theta_t - {\eta \over \sqrt{E[g^2]_t + \epsilon}}g_t$$

Moving average has a property of "forgetting", I mean it decreases past squared gradients when they were calculated long time ago. This property emphasizes recent updates, not the old ones.
Geoffrey Hinton suggests value of $\gamma=0.9$ and $\eta=0.001$

## AdaDelta

This optimization algorithm improves idea on AdaGrad even further.
It also uses Moving Average, but proposes new idea.
Accumulation of Updates (deltas) through the moving average.
Method proposes this because of the following problems:
1) Units mismatch
2) Scaling problem

The update rule for AdaDelta:

$$\Delta \theta_t = {RMS[\Delta\theta]_{t-1} \over RMS[g]_t}g_t$$
$$\theta_{t+1} = \theta_t - \Delta\theta_t$$
Accumulation of Deltas results in:
1) Smarter scaling (our learning rate does not vanish)
2) No need to specify $\eta$.

The update rule with RMS expanded:

$$\Delta \theta_t = {\sqrt{\sum^T_{t=1}\gamma E[\Delta\theta^2]_{t-1} + (1-\gamma) \Delta \theta^2_t + \epsilon}\over \sqrt{\sum^T_{t=1}\gamma E[g^2]_{t-1} + (1-\gamma) g^2_t}+\epsilon}$$
$$\theta_{t+1} = \theta_t - \Delta \theta_t$$

(I'm scared to see how it would look like on GitHub...)
For github:
<img src="https://quicklatex.com/cache3/0d/ql_b82fdb9325aa351562ad5240efe90d0d_l3.png" width=60%>



## Adam

Adam optimizer is the most popular optimizer in Deep Learning

![[Pasted image 20240812150136.png]]

This optimizer combines ideas from RMSprop and Momentum.
It's super efficient and fast and that's cool.
Adam consists 2 moments:
1) mean
2) uncentered variance

Steps of Adam:
1) compute 2 moments
		$m_t = \beta_1 m_{t-1} + (1 - \gamma) g_t$
		$v_t = \beta_2 v_{t-1} + (1-\gamma)g^2_t$

2) Bias correction.
		This step is crucial in Adam optimizer.
		Why?
		We initialize our $m$ and $v$ to 0s, so first values of them will be biased around 0.
		To prevent it we increase their value.
		$\hat m_t = {m_t \over 1-\beta_1^t}$
		$\hat v_t = {v_t \over 1-\beta_2^t}$
		This step is crucial. On first iterations when our $t$ is not big we'll have $\beta_1^t=0.9^t$ where $t$ is not really big (Adam creators recommend $\beta_1 = 0.9$ and $\beta_2 = 0.999$)
		When t is small (from 1 to 5 etc.) we divide our moment by small number increasing it it. However when t is large (100 etc.) We will divide by 1(approximately).
3) Update parameters.
		$\theta_{t+1} = \theta_t - {\eta \over \sqrt{\hat v_t + \epsilon}} \hat m_t$

Let's be honest. Ezy Pezy, but Bias Correction may be confusing for some viewers, so here's an Andrew Ng's explanation, which is part of the Deep Learning Specialization:

[Bias Correction of Exponentially Weighted Averages (C2W2L05)](https://www.youtube.com/watch?v=lWzo8CajF5s)

## Adamax
This is an extension to the original Adam optimization algorithms, which is robust to outliers.
**Outliers** - samples of dataset that differ very much from average sample in dataset. (this is my interpretation of it)
Adamax proposes $l_{\infty}$ norm instead of $l_2$ norm.
Let's clarify the meaning:

$l_2$ - Euclidean distance (square root of sum of squares)
$l_{\infty}$ - Max value in vector.

Update rule for this Algorithm:
$$u_t = max(\beta_2 \cdot v_{t-1}, |g_t|)$$
$$\theta_{t+1} = \theta_t - {\eta \over u_t} \hat m_t$$
We don't need bias correction here for $v_t$, because we're taking maximum by using $l_{\infty}$ norm.
It's robust to outliers because of $u_t$ that picks maximum between previously accumulated gradients and current one.
Outliers usually have big derivative values, so we will divide learning rate by big value (as max will pick big gradient), and update for outlier will be super small.

## Nadam

Nadam is an Adam algorithm with Nesterov Momentum instead of Momentum.
However integration of Nesterov Momentum is harder than it may seem.
It's too exhaustive to update gradient, calculate momentum with updated gradient and update parameters.

Wrong Way:
$$g_t = \nabla J(\theta_t - \gamma m_{t-1})$$
$$m_t = \gamma m_{t-1} + (1-\gamma)\eta g_t$$
$$\theta_{t+1} = \theta_t - m_t$$
Instead, Author of Nadam updates it smarter.

$$g_t = \nabla J(\theta_t)$$
$$m_t = \gamma m_{t-1}+ (1-\gamma) \eta g_t$$
$$\theta_{t+1} = \theta_t - (\gamma m_t + \eta g_t)$$

Note, In original momentum our update is:
$$\theta_{t+1} = \theta_t - (\gamma m_{t-1} + \eta g_t)$$
because $$m_t = \gamma m_{t-1} + \eta g_t$$
We're using $\gamma m_{t-1}$ here, while Dozat's update uses $\gamma m_t$.
That is to say that Nadam uses gradients accumulated till this moment with current gradient.
It gives us an approximation of this "step ahead".

Integrating it into Adam we get the following equation:



Final Update rule:

$$\theta_{t+1} = \theta_t - {\eta \over \sqrt{\hat v_t + e}}({\beta_1m_t \over1-\beta_1^t} + {(1 - \beta_1) g_t \over 1-\beta_1^t})$$

However we can simplify it, because $\beta_1m_{t} \over1-\beta_1^t$ is bias-corrected 

$$\theta_{t+1} = \theta_t - {\eta \over \sqrt{\hat v_t + e}}(\beta_1 \hat m_t + {(1 - \beta_1) g_t \over 1-\beta_1^t})$$
I'll improve this explanation tomorrow!
But, bye for now!

---

[1] - Updates may be too noisy, because of tuning on one particular sample. Data samples may vary very much, what leads us to noisy optimization
[2]-Illustration given by Geoffrey Hinton on his Coursera Class.