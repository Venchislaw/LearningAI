# CNNs Day 1
#DeepLearning  #Medium  #CV
2024-08-19 : 08:47

recap zone:
```
In previous courses of this specialization we learned Multi-Layer Perceptrons in details. Although this architecture is really popular there's a better solution for computer vision problems
```

---

During specialization Andrew explained importance of deep representations (multi-layer structure) on example of image processing, where we detect small lines on the first layer, then detect edges on the second layer, shapes on the third and so on...
It's basically CNNs processing.

We apply operation of **Convolution** with some **kernel**
![[Pasted image 20240819085426.png]]

Note that $*$ is originally convolution symbol, but in python it also means multiplication.
On this screenshot Andrew labeled kernel as filter.
This kernel slides above our data given, multiplies respective elements and sums them up.

![[Pasted image 20240819085847.png]]

Result:

![[Pasted image 20240819090103.png]]

This is an example of Verical Edge detection, and let's see why:
![[Pasted image 20240819090423.png]]
This is a really basic and simple example of how convolution works and may be helpful.
However for me it's not enough, because I

## But what is a convolution?
I gave an explanation in [[convolution]]

21.08.2024
Sorry for this gap, I was busy.

One of the most powerful ideas In computer vision is kernel with weights.
It lets our network create custom and representative filters.

### Padding
In convolution we have problems with dimensions values.
The size of output is always smaller. It's computed using the formula:
$$n - f + 1$$
Where $n$ - size of an image, $f$ - size of a kernel.

Problems:
1) Image decay. We can not apply infinitely many filters, because each of them reduces size of our output image.
2) Pixels on edges are used less, while central kernels are actively used, because of kernel overlapping.

To prevent this image size shrinking we can add some sort of a "border" filled with zeros.

<br><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRxTZKpsJRKk7bYZ_Jeo9oKs5j75XT4U5PVWQ&s">
<br>
With this method we tackle the problem of image shrinking.
This padding doesn't have strong effect on values, because it is filled with zeros.
Our convolution looks like this:<br>
<img src="https://miro.medium.com/v2/resize:fit:1400/1*O06nY1U7zoP4vE5AZEnxKA.gif" width=70%>

<br>
We can choose any padding value $p$. In this example $p=1$

This concept leads us to two types of convolution.
#### Valid Convolution

$$n\times n * f \times f = n-f+1 \times n-f+1$$
This is a standard convolution, we have learned before.

#### Same Convolution
$$n + 2p\times n+2p * f\times f = (n+2p)-f+1\times(n+2p)-f+1$$
From this equation we want $(n+2p)-f+1$ to be $n$

$$n+2p - f + 1=n$$
$$2p-f+1$$
$$p = {f-1\over2}$$
This is the padding value we have to use to preserve the size of an image.
From this formula we can see that we should better have odd value of kernel size.

Typical kernel values are: $3, 5, 7, 9 ...$

In this way we can do the following when building CNN:
- Specify padding by hand
- Use valid convolution ($p=0$)
- Use same convolution ($p = {f-1\over2}$)

### Strides

<br>
<img src="https://miro.medium.com/v2/resize:fit:588/1*BMngs93_rm2_BpJFH2mS0Q.gif" width=30%>
<br>
In all previous examples we took step of 1, but we can specify stepsize (stride) to some other value. 
Formula for the image size:

$${n + 2p - f \over s}+1$$
Note that $p$ can be 0.

In case $n+2p-f \over s$ is not an integer, we should round it down (floor).
Uh, oh... That's it here.

### Convolutions on volume.

In real life we thankfully see our world in colors. Colorful images are represented as 3D matrices. Example shape:
$$(256, 256, 3)$$ That said, we can't use our ordinary 2D kernel anymore. Instead we have to turn it 3D.
Example shape:
$$(3, 3, 3)$$
The rest of an algorithm is identical. Our cubic kernel slides across an entire image and produces a 2D output...
<br>
<img src="https://miro.medium.com/v2/resize:fit:1400/1*hGyDx8VKYPnJqIoCidLzFw.gif" width=50%>
<br>
This may sound like a problem, but who said than we can use only 1 filter?
Why don't we use 2, 3, 10, 999 and stack them together?
In case of $x$ filters for this image and for 3x3 filter we will have output of:
$$(4, 4, x)$$
That's not a even a problem anymore.
Actually it's a huge W!
$x$ is a number of features we're detecting (vertical edges, horizontal edges etc.)
Convolutions over volume are even COOLER!

## Layer of Convolutional Neural Network

In Multi-layer perceptron we had the following layer structure:
$$z^{[1]} = W^{[1]}a^{[0]} + b^{[1]}$$
$$a^{[1]} = g(z^{[1]})$$

In convolutional neural network we do the following:
1) $n$ filters filled with our learnable parameters ($w$)
2) Add bias $b$ to every value of the output (after applying kernel)
3) Activate This output (ReLU etc.)
4) Stack activated outputs

Here's what it does:

![[Pasted image 20240821172817.png]]

Here's an analogy:

![[Pasted image 20240821173030.png]]

#### Notations
Oh blyat... (Oh man...)

$f^{[l]}$ - filter size at layer $l$
$p^{[l]}$ - padding at layer $l$
$s^{[l]}$ - stride at layer $l$
$n_c^{[l]}$ - number of filters

Input:
$n_H^{[l-1]} \times n_W^{[l-1]} \times n^{[l-1]}_c$, where $n_c^{[l-1]}$ - number of channels, $H$ and $W$ - height and width, respectively.

Output:
$n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$, where:

$n^{[l]}_H = floor({n_H^{[l-1]} + 2p^{[l]} - f^{[l]} \over s^{[l]}} + 1)$
$n^{[l]}_W = floor({n_W^{[l-1]} + 2p^{[l]} - f^{[l]} \over s^{[l]}} + 1)$

Each filter is $f^{[l]} \times f^{[l]} \times n_c^{[l-1]}$
number of channels in filter must match number of channels in input.
Activations:
$a^{[l]} -> n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$
$A^{[l]} -> m \times n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$

Weights:
$f^{[l]} \times f^{[l]} \times n_c^{[l-1]} \times n_c^{[l]}$
(reminder: $n_c^{[l]}$ - number of filters in this layer, while $n_c^{[l-1]}$ - number of filters in the previous layer output passed as input to this layer)

Bias:
$n_c^{[l]}$, but in code it's better to use $(1, 1, 1, n_c^{[l]})$ shape.


<br>
<img src="https://media.tenor.com/izxw0H5mkAQAAAAi/meme-realisation.gif" width=5%><img src="https://media.tenor.com/izxw0H5mkAQAAAAi/meme-realisation.gif" width=5%><img src="https://media.tenor.com/izxw0H5mkAQAAAAi/meme-realisation.gif" width=5%><img src="https://media.tenor.com/izxw0H5mkAQAAAAi/meme-realisation.gif" width=5%><img src="https://media.tenor.com/izxw0H5mkAQAAAAi/meme-realisation.gif" width=5%><img src="https://media.tenor.com/izxw0H5mkAQAAAAi/meme-realisation.gif" width=5%>


<br>