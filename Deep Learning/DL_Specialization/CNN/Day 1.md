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