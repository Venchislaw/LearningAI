This is a digital Note taken in Obsidian md format to show step by step Neural Network implementation from scratch, including all math needed, different techniques and code.
**Plan:**
- What is Neural Network
- Math of Neural Network
- Improvement Techniques
- Vectorization
- Python Code
- Conclusion

Author:<br>
<img src="https://avatars.githubusercontent.com/u/171679851?v=4" width=30%>
Venchislaw - [Github](https://github.com/Venchislaw)
MIT License

---
## What is Neural Network

Neural network is first of all - mathematical abstraction. It's a complex crazy function that takes inputs, and produces outputs through crazy computations chain. Word math may be confusing here, because a lot of people hate math and prefer not touching it. However math is that what makes neural networks as complex as they are simple.
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
$b$ - y-intersect

Graph of such function looks like this:

<img src="https://cdn-academy.pressidium.com/academy/wp-content/uploads/2021/12/key-features-of-linear-function-graphs-2.png" width =30%>

Now let's look at our neural network:<br>
<img src="https://res.cloudinary.com/practicaldev/image/fetch/s--2UPg0Z-6--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://imgur.com/KeIJEYy.jpeg" width=30%>
It consists of *circles*. Those circles are **neurons**
Each neuron does this linear computation on inputs passed to it:

<img src="https://www.allaboutcircuits.com/uploads/articles/how-to-train-a-basic-perceptron-neural-network_rk_aac_image1.jpg" width=30%>

We know that this $\sum$ sign means summation, so this summation (for neuron above) looks like the following:
$$\hat y = w_0x + w_1y + w_2z + b$$
It's absolutely the same, but with more dimensions.
Btw this particular image is shitty, as inputs are denoted as $x$, $y$ and $z$
I'll use the following notation:
$x$ - input
$y$ - output

And we basically learn all these parameters:
$w_0$ $w_1$ $w_2$ $b$

But it's not enough...
In forward pass this calculation: 
$$w_0x_0 + w_1x_1 + w_2x_2 + b$$
Is denoted with $z$ letter. Because we need to *activate* our neuron.
So far we know that neural network consists of simple neurons (in this case they are linear units). But linear functions stacked together form ... Linear function.
In order to fix it we have many cool math functions to make our outputs $z$ non-linear.
It's done to form shapes of input data for good fit.
These neurons stacked together vertically construct *layers* that are colored and denoted on image.

### Activation Functions

**ReLU**

This activation function is the most common because of its simplicity mixed with efficiency.
This function is called Rectified Linear Unit because it's nearly linear:
<img src="https://media.licdn.com/dms/image/D4D12AQEDy0qH_OQgjg/article-cover_image-shrink_600_2000/0/1658772158369?e=2147483647&v=beta&t=IepDZxgXRKJzHXGZQTwAIcUno5CUOeDQzygPk19yRBA" width=50%>
Or simply
$$\begin{cases} 
      z & z\gt 0 \\
      0 & z\leq 0
   \end{cases}
$$

**Sigmoid**

This activation function is great for *binary classification problems* where we classify sample as positive\negative, cat\dog, car\plane etc. However this function is commonly applied only to output layers.
<img src="https://raw.githubusercontent.com/Codecademy/docs/main/media/sigmoid-function.png" width=50%>

Formula is trickier:
$$1 \over {1 + e^{-z}}$$
Where $z$ is our calculated z.
It's also known for *probabilistic output* i.e function returns value from 0 to 1, which is considered as probability of getting 1 class.

**Tanh**

Similar to sigmoid (actually it's just a shifted sigmoid)

<img src="https://images.squarespace-cdn.com/content/v1/5acbdd3a25bf024c12f4c8b4/1524687495762-MQLVJGP4I57NT34XXTF4/TanhFunction.jpg" width=50%>
$$e^z - e^{-z} \over {e^z + e^{-z}}$$
tanh is much better for *hidden layers* (layers between input layer and output layer)