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
- Conclusion\

Enjoy!

---

Parameter tuning based on the Loss Function results is the key part of Deep Learning that gives our models ability to learn. Picking a good optimizer can be really tricky process full of confusion, so this article will describe all main optimizers in Deep Learning. This article was inspired by the following paper: [An overview of gradient descent optimization algorithms - Sebastian Ruder](https://arxiv.org/pdf/1609.04747), however in this article I try to cover topic wider, introducing all formulas and their effects.



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
<img src="https://www.researchgate.net/publication/365820889/figure/fig3/AS:11431281103439668@1669691987354/Nesterov-momentum-first-makes-a-big-jump-in-the-direction-of-the-previously-accumulated.ppm" widht=30%>
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



---

[1] - Updates may be too noisy, because of tuning on one particular sample. Data samples may vary very much, what leads us to noisy optimization
[2]-Illustration given by Geoffrey Hinton on his Coursera Class.