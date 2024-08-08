This is a digital Note taken in Obsidian md format to show step by step Neural Network implementation from scratch, including all math needed, different techniques and code.
**Plan:**
- What is Neural Network
- Forward Prop
- Let's build...
- Backward Prop
- Improvement Techniques
- Vectorization
- Python Code
- Conclusion

Author:<br>
<img src="https://avatars.githubusercontent.com/u/171679851?v=4" width=30% style="border-radius:50%;">

Venchislaw - [Github](https://github.com/Venchislaw)
MIT License

---
## What is Neural Network

Neural network is first of all - mathematical abstraction. It's a complex crazy function that takes inputs, and produces outputs through crazy computations chain. Word "math" may be confusing here, because a lot of people hate math and prefer not touching it. However math is that what makes neural networks as complex as they are simple.
Neural Network probably has nothing to do with our Human brain. Neurons in our brain work much harder than artificial neuron. What makes this name is an associaton with biological brain.

Biological neuron takes some inputs from other neurons and produces some outputs, that are passed to other neurons as inputs. It all forms kind of Network.
Now We have some brief intuition about what is called Neural Network, so let's move on to other part.

---
## Math
I've already mentioned, that math makes topic of Deep Learning as hard as simple.
The thing is that we may face some huge formulas and a lot of greek letters like shown below:
$$\sum a \beta \gamma \sigma \prod$$
However knowledge of notations makes it all easier:
$\sum$ - sum (works like sum() function in python)
```python
sum([1, 2, 3])  # 6
```
$\prod$ - product (there is no python function for this, but code below represents this notation)
```python
num = [1, 2, 3]
product = 1
for num in nums:
	product *= num

print(product)  # 6
```

### Forward Propagation
Neural networks are super cool because they can learn any pattern from any data (nearly)
But how do they do this?
We train our model on given data, and training consists of 2 parts:
- Forward propagation
- Backward propagation

Forward propagation makes predictions (and we will see how) while Backward propagation tunes *parameters* of model to minimize *loss* and be more precise.

**Analogy**
Imagine you are cooking some delicious dish for your guests!
You value your friends, so you want to cook the best dish.
However you don't have recipe, but have a lot of *parameters* of dish:
- Level of saltiness
- Meat Type
- Sauce used
and so on...
You can try cooking the dish over and over again, giving every generation of dish to your friends and asking for their feedback. 
When they say: "It's too salty" - you should add less salt to the next generation.

**Parameters**
Okay, but what are the parameters of model?
Of course, for better understanding it does worth learning ML first, but let's skip this part.

From school you may be familiar with Linear function formula:
$$y = wx + b$$
Where:
$y$ - output
$w$ - slope
$x$ - input
$b$ - y-intercept

Graph of such function looks like this:

<img src="https://cdn-academy.pressidium.com/academy/wp-content/uploads/2021/12/key-features-of-linear-function-graphs-2.png" width =30%>

Now let's look at our neural network:
(this is only 1 type of neural nets known as Multi Layer Perceptron)
<br>

<img src="https://res.cloudinary.com/practicaldev/image/fetch/s--2UPg0Z-6--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://imgur.com/KeIJEYy.jpeg" width=30%><br>

It consists of *circles*. Those circles are **neurons**
Each neuron does this linear computation on inputs passed to it:

<img src="https://www.allaboutcircuits.com/uploads/articles/how-to-train-a-basic-perceptron-neural-network_rk_aac_image1.jpg" width=30%>

We know that this $\sum$ sign means summation, so this summation (for neuron above) looks like the following:
$$\hat y = w_0x + w_1y + w_2z + b$$
It's absolutely the same, but with more dimensions.
Btw this particular image is shitty, as inputs are denoted as $x$, $y$ and $z$
I'll use the following notation:
$x_i$ - input
$y_i$ - output

Parameters in this case are:
$w_0$ $w_1$ $w_2$ $b$

But it's not enough...
In forward pass this calculation: 
$$w_0x_0 + w_1x_1 + w_2x_2 + b$$
Is denoted with $z$ letter. Because we need to *activate* our neuron.
And also because $\hat y$ is used to denote final Neural Net output.
So far we know that neural network consists of simple neurons (in this case they are linear units). But linear functions stacked together form ... Linear function.
It's bad because it doesn't worth it. It's just a complex linear function.
In order to fix it we have many cool math functions to make our outputs $z$ non-linear.
It's done to fit input data well.
These neurons stacked together vertically construct *layers* that are colored and denoted on image.

### Activation Functions

**ReLU**

This activation function is the most common because of its simplicity mixed with efficiency.
This function is called Rectified Linear Unit because it's nearly linear:<br>
<img src="https://media.licdn.com/dms/image/D4D12AQEDy0qH_OQgjg/article-cover_image-shrink_600_2000/0/1658772158369?e=2147483647&v=beta&t=IepDZxgXRKJzHXGZQTwAIcUno5CUOeDQzygPk19yRBA" width=50%>
<br>Or simply<br>
$$\begin{cases} 
      z & z\gt 0 \\
      0 & z\leq 0
   \end{cases}
$$

**Sigmoid**

This activation function is great for *binary classification problems* where we classify sample as positive\negative, cat\dog, car\plane etc. However this function is commonly applied only to output layers.<br>
<img src="https://raw.githubusercontent.com/Codecademy/docs/main/media/sigmoid-function.png" width=50%>
<br>
Formula is trickier:
$$1 \over {1 + e^{-z}}$$
Where $z$ is our calculated z.
It's also known for *probabilistic output* i.e function returns value from 0 to 1, which is considered as probability of getting 1 class.

**Tanh**

Similar to sigmoid (actually it's just a shifted sigmoid)<br>

<img src="https://images.squarespace-cdn.com/content/v1/5acbdd3a25bf024c12f4c8b4/1524687495762-MQLVJGP4I57NT34XXTF4/TanhFunction.jpg" width=50%><br>

$$e^z - e^{-z} \over {e^z + e^{-z}}$$
tanh is much better for *hidden layers* (layers between input layer and output layer) in comparison to sigmoid, but still worse than relu.


**Softmax**

Hero of the day! Okay, this one is really interesting as it introduces Multi-class classification.
Sigmoid works only for binary classification, where we have 2 classes. Here it's different

<img src="https://www.researchgate.net/publication/348703101/figure/fig5/AS:983057658040324@1611390618742/Graphic-representation-of-the-softmax-activation-function.ppm" width=50%><br>
This one is also considered to be a **general case of sigmoid**
Formula here is pretty simple:

$$e^{z^{[L]}} \over \sum^{n_x}_{i=1}e^{z^{[L]}_i}$$
It consists of 2 simple steps:
1) Exponentiate
2) Normalize (such that all categories sum up to 1)
Output here is also a probability where $\hat y[i]$ = probability of getting $i$ class.

Note: Our original input data $y$ is presented in form of **hardmax**:
$$y = \begin{bmatrix}  
0\\  
1\\
0
\end{bmatrix}$$
Onehot-encoded data where instead of label to be $y=1$ we have a vector, where 1st element (count from 0) is equal to 1.

## Putting it all together
Okay, now let's discuss our forward propagation on professional level!
We already know that we do the following computations:
$$z = w_0x_0 + w_1x_1 + ... + w_nx_n + b$$
where n - number of features passed as input
$$a = g(z)$$
$a$ - activated z output
$g$ - activation function

And we do it on some layer $L$ , so let's add notation:
$$z^{[L]} = w_0^{[L]}x_0^{[L]} + w_1^{[L]}x_1^{[L]} + ... + w_n^{[L]}x_n^{[L]} + b^{[L]}$$

$$a^{[L]} = g^{[L]}(z^{[L]})$$
Attentive reader may take a note:
On layer $L+1$ we'll use output $a$ of layer $L$
So our formula will change a little:
$$z^{[L+1]} = w_0^{[L+1]}a_0^{[L]} + w_1^{[L+1]}a_1^{[L]} + ... + w_n^{[L+1]}a_n^{[L]} + b^{[L+1]}$$
And so on...
But it's time consuming to write this equation in such form.
Thankfully there is math!
We can stack our weights and inputs as vectors (for 1 neuron they are vector*, but for layer it's matrix*)
And we'll have a *dot product*!

$$z = wx + b$$
BAM!!!
Easier!

Now let's put it all together to form a calculation for layer!

$$\begin{bmatrix}  
w_{1_1} & w_{2_1} & w_{3_1}\\  
w_{1_2} & w_{2_2} & w_{3_2}\\
w_{nn_x} & w_{nn_x} & w_{nn_x}
\end{bmatrix}$$

$$\begin{bmatrix}  
b_{1}\\  
b_{2}\\
b_{n}
\end{bmatrix}$$
and our input is also a matrix.
$$X^T = \begin{bmatrix}  
x_{1_1} & w_{2_1} & w_{m_1}\\  
w_{1_2} & w_{2_2} & w_{m_2}\\
w_{1_n} & w_{2_n} & w_{m_n}
\end{bmatrix}$$
Note: Matrix is transposed, meaning it's not convenient form of dataset where each row represents a sample and each column a feature. It's rotated, so Each column is a sample with row features.
We did it because of dimensions that i didn't mention.

Dot product works only when "neighboring" inner dimensions are equal:
$(545, 5) (5, 348)$ etc.
Result of dot product of shapes is a shape of "outer" dimensions:
$(545, 348)$

**Is math still being complex to you?**
<br>
<img src="https://pbs.twimg.com/media/Fr_gyvQX0AAJHVO.jpg" width=20%>
<br>


Resulting computation:
$$A^{[L]} = g(WX^T+b)$$

We do it for each layer and pass output $A^{[L]}$ as input to the layer $A^{[L+1]}$

*  Layer and matrix are concepts from Linear Algebra that are out of bound of this material, but loosely speaking here's python representation
```python
vector = [1, 2, 3]  # array

# array of arrays
matrix = [[1, 2, 3],
		  [4, 5, 6],
		  [7, 8, 9]]
		  
```

If you have any questions after this part:<br>
- [3Blue1Brown Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Forward Propagation](https://www.youtube.com/watch?v=a8i2eJin0lY)

### Goal

Let's set some goal to build it here.
Let's build a simple neural network for iris classification by numerical features.
<br>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Iris_%28plant%29.jpg/800px-Iris_%28plant%29.jpg" width=30%>
<br>
[Dataset Link](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)

Our neural Network architecture:<br>
<img src="https://i.ibb.co/nPwvMPw/Screenshot-from-2024-08-07-15-11-27.png" width=50%><br>
It's just random architecture, but anyways...
It's simple, but it's enough for our educational purposes

We have multi-class classification problem. So as mentioned earlier we will use **softmax**.
As activations between layers we'll use ReLU
Okay, we have *architecture* and *activation functions*
We're ready!
Or are we?

In example with dish I mentioned feedback.
Same here. We need to know how precise our model is.
<br><img src="https://images.hindustantimes.com/img/2024/08/01/1600x900/turkish_shooter_memes__1722511027152_1722511027296.jpeg" width=30%><br>

## Loss Function (and cost function)

Loss function is our "feedback". It outputs small value if we're precise and big value when we're not.
Input of Loss Function is a pair of predicted value and expected value:

$$L(\hat y, y)$$
There are many different loss functions for different tasks, but here we will use categorical crossentropy loss:

$$- \sum_{i=1}^c y_i log(\hat y_i)$$
c - number of classes in data (3 in our case).
Log returns 0 if $\hat y_i$ = 1, but when it's close to 0 (log of 0 is -$\infty$ ).
We multiply by $y_i$ in order to multiply $-\infty$ by 0 (if our $y_i$ =0 and $\hat y_i$ =0 our model didn't make mistake, so we multiply it by 0), but when our prediction is some big negative number ($-\infty$ for example), meaning our model predicted 0, but $y=1$ we will add it to our loss. We have negative sign as $log$ in all these cases returns negative values.

But what is cost function then?

It's simply the sum of loss function outputs for all samples:

$$J(w_0, w_1, ..., b_n) = {1 \over m} \sum^{m}_{i=1} L(y_i, \hat y_i)$$
Okay, we get some feedback... What do we do next?
Oh... We're closer to backprop...
We need to tweak our parameters to make $J$ smaller.

Experienced with math people understand that it is a problem of optimization where we minimize/maximize some function.
The most popular algorithm in Machine Learning for parameters optimization is called
**Gradient Descent**

By far you may think that gradient is a mix of beautiful colors, but I'll change your mind...
<br><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXZsC6D_I-pe_GEnklMlSlM_9jEkJYiu6tWQ&s" width=30%> <br>


---

## Backward Propagation
<br>
<img src="https://i.kym-cdn.com/photos/images/newsfeed/001/623/463/d90.jpg" width=21% ><br>
Aw man... How many people broke their brains trying to understand it...
Don't worry btw.

Math again:

## Derivatives
If you are familiar with concept of derivative and can calculate the following derivative:

$$f(x) = x^2$$
$$f' - ?$$
Feel free to skip this derivative introduction.
Look at this beautiful function:<br>
<img src="https://saylordotorg.github.io/text_intermediate-algebra/section_09/f7c27c7d79c9141d0731362a4554caa7.png" width=70%><br>
This is a Quadratic Parabola.
We know how it produces outputs, but here's an interesting question:
```
Can we describe how fast this function changes, and if we can
How?
```
It may be confusing, but mathematicians came up with an Idea of Derivative!
This beauty gives us number that represents how fast our function changes in particular point of graph.

Algorithm is really simple
Bear with me!
1. Take some x value you are interested in and calculate y for it
2. Add some tiny tiny value $\epsilon$ to your x
3. Calculate y1 for your updated x

$$slope = {{y1 - y} \over \epsilon}$$
If you grasp the idea, by thinking deeper it's literally what it is!
In complex functions with many variables we can find derivatives *with respect to some variable*

for example:

$${d \over da}2ac + 3b = 2c$$
We simply "close" our eyes on parts of function where a is not used. We worry only about the part where a is multiplied by something.
There are also many rules of derivation, but they're out of bounds of this tutorial.

## Usage of Derivatives:

With derivatives we literally by definition understand how change in variable affects result.
You might have already understood how we'll use derivatives.

$$dL \over dW$$
$$dL \over db$$
Our gradient is positive when it goes up and it's negative when we go downwards.<br>
<img src="https://miro.medium.com/v2/resize:fit:1400/1*2aS_8T7-f5gkoE-3gA9JlA.png" width=60%>
We need to move "right" when gradient is negative (this said we need to increase parameter if it's gradient is negative), otherwise move left(decrease parameter value).

That would look like this:

$$W = W - {dL \over dW}$$
$$b = b - {dL \over db}$$
But on practice such updates are too rapid and fast. We need some coefficient (multiplier) to slow down updates a little bit.
So that said our update formula is defined as:

$$W = W - a{dL \over dW}$$
$$b = b - a{dL \over db}$$

Where $a$ -*learning rate* (value [0, 1] that limits speed of updates)

Now main question is how do we find $dL\over db$

## Computation Graph

Our neural networks may be pretty big and complex, so to ease computations of derivatives I need to introduce an Idea of **computation graph**.

Imagine some function $f$ defined as:
$$f(x, y, z) = x(2y+z)$$
When plugging in some inputs and calculating output we do some ordered operations:
1) a=2y
2) b=a+z
3) c=x*b

Computation graph may look like this:
<br>
<img src="https://i.postimg.cc/VsSFHxtq/Screenshot-from-2024-08-08-16-00-07.png" width=60%>
<br>
From here we can calculate derivatives w.r.t some variable (w.r.t - with respect to)
From this image we can also understand the idea of **Chain Rule**
For example let's try calculating
$$dc\over dz$$
It may be confusing, but let's thing logically.
We can split this problem on sub-problems.

1) $dc\over db$
2) $db \over dz$
3) ${dc\over dz} = {dc\over db}{db\over dz}$
4) Enjoy results!
We find out how change in b affects c, and then calculate how much b depends on c.
And we multiply it starting from the end!
Ezy peezy!

## Computation Graph of our Network

<br>
<img src="https://i.postimg.cc/Z5qnwSBZ/Screenshot-from-2024-08-08-16-14-05.png" width=100%>
