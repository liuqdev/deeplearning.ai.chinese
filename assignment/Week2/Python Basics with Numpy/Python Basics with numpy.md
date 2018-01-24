
# Numpy基本操作

欢迎来到你的第一个深度学习专业(可选)编程训练。这个任务中你将学到：

- 学会如何使用numpy.

- 动手实现一些基础和兴的深度学习函数，例如softmax, sigmoid, dsigmoid等

- 学会如何处理数据，比如正则化输入和重设输入图像大小.

- 意识到向量化的重要性.

- 理解Python的广播机制是如何运行的.

以上就是为你进行准备的任务.花费一些功夫来完成并且确保通过训练来得到理想的输出结果. 在一些编程块中,你会发现"#一系列函数: 函数名称"的备注,请不要修改.在你完成练习后，提交你的工作并检验结果。正确率达到70%以上就算作通过.祝你好运.:-)!

# Python的numpy基础操作 v3版本 (可选的练习)


欢迎来到你的第一个任务.假如你之前从未使用过python的话，通过这个练习你将对python有一个初读的映象，并且还会帮助你熟悉你所需要的函数.

**说明:**
- 本练习使用的是Python3
- 除非明确地指出要使用到for循环或者while循环，否则尽量不要使用
- 不要修改在某些小单元中的(# GRADED FUNCTION [function name])注释.变更之后不保证能够正确地给你的程序判分,另外,每个这样的评论中至多写一个函数
- 编写你自己的函数之后,运行你的程序,来验证自己的结果

**通过本次练习你将:**
- 能够使用iPython (Jupyter)Notebooks
- 能够使用numpy函数和numpy内置的矩阵/向量操作符
- 理解"广播"的机制
- 能够编写向量化程序

让我们开始吧！

## 关于ipython (Jupyter) Notebooks

Ipython (Jupyter) Notebooks是一款嵌入到浏览器当中的交互式编程环境.你将在这门课程中使用它.你只需要在### 代码开始 ### 和 ### 代码结束 ###注释中间来编写代码.写完你的代码之后,你可以按住`Shift`+`Enter`或者点击上方的`Run Cell(播放标志图案)`按钮来运行代码.

当你看到"(≈ X 行代码)"的注释，表明此处大约需要编写多少行程序,实际可稍长或稍短.

**Exercise**: 试着运行下面的示例，从`Hello World`开始.


```python
### 代码开始 ### (≈ 1 行代码)
test = "Hello World"
### 代码结束 ###
```


```python
print ("test: "+ test)
```

    test: Hello World
    

**期望输出**:
test: Hello World

<font color='green'>
**这个示例旨在告诉我们**
- 按住`SHIFT+ENTER`(或者点击上方的`Run Cell`)来运行
- 使用Python3在特定位置写下你的代码
- 不要修改特定区域之外的代码  


## 1 - 使用numpy来编写基本的函数 ##

Numpy是Python中用于科学计算的主要几个包之一.在它背后有强大的而活跃的(www.numpy.org)社区在支持.在这个联系中，你会学到numpy中几个关键的函数，例如np.exp, np.log, 以及np.reshape. 你需要掌握这些函数的使用，因为在之后的练习中你还会用到它们.

### 1.1 - sigmoid 函数， np.exp() ### 

在学会使用np.exp()之前，你会使用math.exp()来动手实现sigmoid函数.之后你就会理解为什么np.exp()性能比math.exp()要好.

**练习**: 编写一个函数,返回一个实数的sigmoid值,其中的指数函数使用,math.exp().

**回顾**:
$sigmoid(x) = \frac{1}{1+e^{-x}}$有时候也叫做logistic function(逻辑函数).它是一种典型的非线性函数,不仅应用于机器学习,同时也应用于深度学习中.

<img src="images/Sigmoid.png" style="width:500px;height:228px;">

当你引用一个特定的库中的函数时,要预先声明`package_name.function()`.


```python
# 函数: basic_sigmoid

import math

def basic_sigmoid(x):
    """
    计算一个实数的sigmoid值
    
    参数:
    x -- 一个标量
    
    返回:
    s -- sigmoid(x)
    
    """
    ### 代码开始 ### (≈ 1 行代码)
    s = 1.0/(1+math.exp(-x))
    ### 代码结束 ###
    
    return s
```


```python
basic_sigmoid(3)
```




    0.9525741268224334



**期望输出**: 
<table style = "width:40%">
    <tr>
    <td>** basic_sigmoid(3) **</td> 
        <td>0.9525741268224334 </td> 
    </tr>

</table>

实际上我们在深度学习中很少使用"math"库,因为其输入要保证是实数.在深度学习中我们使用矩阵以及向量,这就是为什么numpy更加实用了.


```python
### 在深度学习中,我们使用numpy来代替math中的原因之一 ###
x = [1, 2, 3]
print(type(x))
basic_sigmoid(x) # 运行之后回出现一个错误,因为这里x是一个向量
```

    <class 'list'>
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-4-457c6d5855eb> in <module>()
          2 x = [1, 2, 3]
          3 print(type(x))
    ----> 4 basic_sigmoid(x) # 运行之后回出现一个错误,因为这里x是一个向量
    

    <ipython-input-1-0d59d93f9ecb> in basic_sigmoid(x)
         15     """
         16     ### 代码开始 ### (≈ 1 行代码)
    ---> 17     s = 1.0/(1+math.exp(-x))
         18     ### 代码结束 ###
         19 
    

    TypeError: bad operand type for unary -: 'list'


假如$ x=(x_1, x_2, ..., x_n)$是一个行向量, $np.exp(x)$会对x中每一个元素x求出其指数函数值.因此,输出结果会是:$np.exp(x) = (e^{x_1}, e^{x_2}, ..., e^{x_n})$


```python
import numpy as np

# np.exp 使用示例
x = np.array([1, 2, 3])
print(np.exp(x)) # 结果为 (exp(1)), exp(2), exp(3))
```

    [  2.71828183   7.3890561   20.08553692]
    

此外,假如x是一个向量,形如$s = x+3$或者$s = \frac {1}{x}$会输出和输入x大小一致的结果向量.


```python
# 向量操作示例
x = np.array([1, 2, 3])
print(x+3)
```

    [4 5 6]
    

在任何时候,对于numpy中的函数有疑问,都可以去其[官方文档](https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.exp.html)中寻找答案.

你也可以在notebook中的任何一个cell中编写如`np.exp?`来快速查看文档中有关的此内容.

**练习**:使用numpy来实现sigmoid函数

**回顾**:x 可以是一个实数,一个向量,或者一个矩阵.在numpy中我们使用numpy arrays 来表示这些数据结构.暂时不用深入了解.

$$ \text{For } x \in \mathbb{R}^n \text{,     } sigmoid(x) = sigmoid\begin{pmatrix}
    x_1  \\
    x_2  \\
    ...  \\
    x_n  \\
\end{pmatrix} = \begin{pmatrix}
    \frac{1}{1+e^{-x_1}}  \\
    \frac{1}{1+e^{-x_2}}  \\
    ...  \\
    \frac{1}{1+e^{-x_n}}  \\
\end{pmatrix}\tag{1} $$


```python
# 函数: sigmoid

import numpy as np # 书写这句话表示,你可以使用np.function() 代替 numpy.function()来使用numpy中的函数

def sigmoid(x):
    """
    计算x的sigmoid
    
    参数:
    x -- 一个标量,或者任意大小的numpy array
    
    返回:
    s -- sigmoid(x)
    """
    
    ### 代码开始 ### (≈ 1 行代码)
    s = 1.0/(1+np.exp(-x))
    ### 代码结束 ###
    
    return s


```


```python
x = np.array([1, 2, 3])
sigmoid(x)
```




    array([ 0.73105858,  0.88079708,  0.95257413])



**期望输出**: 
<table>
    <tr> 
        <td> **sigmoid([1,2,3])**</td> 
        <td> array([ 0.73105858,  0.88079708,  0.95257413]) </td> 
    </tr>
</table> 


### 1.2 - Sigmoid 梯度

我们需要使用后向传播(backpropagation),计算梯度来优化损失函数(loss functions).下面开始编写你的第一个梯度函数.

**练习**:实现函数 sigmoid_grad() 来计算相对于其输入x的梯度.其梯度计算公式为$$sigmoid\_derivative(x) = \sigma'(x) = \sigma(x)(1 - \sigma(x))\tag{2}$$

编写这个函数通常需要两步:
1. 设s为x的sigmoid, 你也许会发现sigmoid(x)函数十分有用.
2. 计算$\sigma'(x)=s(1-s)$


```python
# 函数: sigmoid_derivative

def sigmoid_derivative(x):
    """
    对于输入x计算其梯度(有时候也叫坡度或者导数).你可以将sigmoid函数的结果存储到变量中,然后使用其来计算梯度.
    
    参数:
    x -- 一个标量或者numpy array
    
    返回:
    ds -- 计算得到的梯度
    """
    
    ### 代码开始 ### (≈ 2 行代码)
    s = 1/(1+np.exp(-x))
    ds = s*(1-s)
    ### 代码结束 ###
    
    return ds
```


```python
x = np.array([1, 2, 3])
print("sigmoid_derivative(x)= "+ str(sigmoid_derivative(x)))
```

    sigmoid_derivative(x)= [ 0.19661193  0.10499359  0.04517666]
    

**期望输出**: 


<table>
    <tr> 
        <td> **sigmoid_derivative([1,2,3])**</td> 
        <td> [ 0.19661193  0.10499359  0.04517666] </td> 
    </tr>
</table> 



### 1.3 - 重塑array

深度学习中经常用到两个函数 [np.shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) 和 [np.reshape()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html). 
- X.shape 来获得一个矩阵或者向量X的形状大小(维度值)
- X.reshape(...) 将X重塑为其他维度值

举例子,在计算机科学中,一幅图像表示为一个三维 array,形状为$(length, height, depth = 3)$.但是当你将其作为一个算法的输入时,需要将其重塑为一个形状为$(length*height, 3)$的向量.换句话说,我们展开,或者说重塑了这张图像,将其从一个三维的array转换为一个一维的向量.

<img src="images/image2vector_kiank.png" style="width:500px;height:300;">

**练习**:实现`image2vector()`,输入参数形状为 (length, heigh, 3) 返回一个向量,形状为 (length\*height\*3, 1).
1). 例如,如果你想把一个形状为(a, b, c)的array v转换为一个形状为(a*b, c)的向量.你可能这样:
```python
v = v.reshape((v.shape[0]*v.shape[1], v.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
```
- 不要将图像的维度数定为常量且不能修改.可以使用`image.shape[0]`来查看你所需要的哪一维的大小.



```python
# 函数: image2vector
def image2vector(image):
    """
    参数:
    image -- 形状为(length, height, depth)的numpy array
    
    返回:
    v -- 形状为(length*height*depth, 1)的向量
    """
    
    ### 代码开始 ### (≈ 1 行代码)
    v = image.reshape( image.shape[0]* image.shape[1]*image.shape[2], 1)
    ### 代码结束 ###
    
    return v
```


```python
# 下面给出3*3*2的array, 典型的图像时 (num_px_x, num_px_y, 3)其中的3代表RGB的值
# 从外到内,最外层代表通道,然后时图像高度,接着时图像宽度
image = np.array([
       [[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print ("image2vector(image) = " + str(image2vector(image)))
```

    image2vector(image) = [[ 0.67826139]
     [ 0.29380381]
     [ 0.90714982]
     [ 0.52835647]
     [ 0.4215251 ]
     [ 0.45017551]
     [ 0.92814219]
     [ 0.96677647]
     [ 0.85304703]
     [ 0.52351845]
     [ 0.19981397]
     [ 0.27417313]
     [ 0.60659855]
     [ 0.00533165]
     [ 0.10820313]
     [ 0.49978937]
     [ 0.34144279]
     [ 0.94630077]]
    

**期望输出**: 


<table style="width:100%">
     <tr> 
       <td> **image2vector(image)** </td> 
       <td> [[ 0.67826139]
 [ 0.29380381]
 [ 0.90714982]
 [ 0.52835647]
 [ 0.4215251 ]
 [ 0.45017551]
 [ 0.92814219]
 [ 0.96677647]
 [ 0.85304703]
 [ 0.52351845]
 [ 0.19981397]
 [ 0.27417313]
 [ 0.60659855]
 [ 0.00533165]
 [ 0.10820313]
 [ 0.49978937]
 [ 0.34144279]
 [ 0.94630077]]</td> 
     </tr>
    
   
</table>

### 1.4 - 正则化

在机器学习和深度学习中一种常用的技术是正则化我们的数据.正则化一般会是计算结果更好,原因是在正则化之后,梯度下降会收敛得更快一些.这里正则化指的是将 x 正则化为$ \frac {x}{\| x \|}$(x 的没行向量除以其范数).

例如,假如 $$x= 
\begin{bmatrix}
    0 & 3 & 4 \\
    2 & 6 & 4 \\
\end{bmatrix}\tag{3}$$ 然后 $$\| x\| = np.linalg.norm(x, axis = 1, keepdims = True) = \begin{bmatrix}
    5 \\
    \sqrt{56} \\
\end{bmatrix}\tag{4} $$ 并且 $$ x\_normalized = \frac{x}{\| x\|} = \begin{bmatrix}
    0 & \frac{3}{5} & \frac{4}{5} \\
    \frac{2}{\sqrt{56}} & \frac{6}{\sqrt{56}} & \frac{4}{\sqrt{56}} \\
\end{bmatrix}\tag{5}$$ 注意你可以对不同形状的矩阵进行除法操作并且也能成功运行:这是numpy中的广播机制,在后面的章节中会进一步介绍.

**练习**:实现`normalizeRows()`来正则化矩阵的每一行.在对矩阵x应用这首函数时,x的每一行应该一个单位向量(及长度为1).



```python
# 函数: normalizeRows

def normalizeRows(x):
    """
    实现一个函数,来正则化输入矩阵x的每一行(得到单位向量)
    
    参数:
    x -- 形状为 (n, m)的numpy matrix
    
    返回:
    x -- (按行)进行正则化后的numpy matrix
    """
    ### 代码开始 ### (≈ 2 行代码)
    # 计算 x_norm 作为 x 的二范数. 使用 np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, ord=2, keepdims=True)
    print("x_norm shape: ",x_norm.shape)
    # x 除以其范数
    x = x/x_norm
    ### 代码结束 ###
    
    return x
```


```python
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("x shape: ",x.shape)
print("normalizeRows(x) = " + str(normalizeRows(x)))
```

    x shape:  (2, 3)
    x_norm shape:  (2, 1)
    normalizeRows(x) = [[ 0.          0.6         0.8       ]
     [ 0.13736056  0.82416338  0.54944226]]
    

**期望输出**: 

<table style="width:60%">

     <tr> 
       <td> **normalizeRows(x)** </td> 
       <td> [[ 0.          0.6         0.8       ]
 [ 0.13736056  0.82416338  0.54944226]]</td> 
     </tr>
    
   
</table>

**注意**:
在`normalizeRows`中,你可以尝试输入x和x_norm的大小,观察中间结果,能够发现他们的大小并不一致.我们计算出了输入矩阵x每一行的范数,范数矩阵x_norm的行数和x的行数相当,但是只有一列.那么 x 除以 x_norm 时,为什么能够生效呢?这就是前面所讲的广播机制,我们将在这里开始讲解.

### 1.5 - 广播机制和softmax函数

"广播机制(broadcasting)"时numpy中一个重要的概念.尤其当我们处理两个不同形状的array的时候.关于广播机制的更多细节,你可以阅读器[官方文档](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

**练习**:使用numpy来实现softmax函数.你可以将softmax视作一种正则化的函数,当我们的算法需要进行二分类或者更多分类的时候使用.关于softmax的更过介绍,在后面的课程中会介绍的到.

**说明**:
- $ \text{对于 } x \in \mathbb{R}^{1\times n} \text{,     } softmax(x) = softmax(\begin{bmatrix}
    x_1  &&
    x_2 &&
    ...  &&
    x_n  
\end{bmatrix}) = \begin{bmatrix}
     \frac{e^{x_1}}{\sum_{j}e^{x_j}}  &&
    \frac{e^{x_2}}{\sum_{j}e^{x_j}}  &&
    ...  &&
    \frac{e^{x_n}}{\sum_{j}e^{x_j}} 
\end{bmatrix} $ 

- $\text{对于一个矩阵 } x \in \mathbb{R}^{m \times n} \text{,  $x_{ij}$ 代表 $x$ 中 $i^{th}$ 行, $j^{th}$ 列元素, 因此有: }$  $$softmax(x) = softmax\begin{bmatrix}
    x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
    x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{m1} & x_{m2} & x_{m3} & \dots  & x_{mn}
\end{bmatrix} = \begin{bmatrix}
    \frac{e^{x_{11}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{12}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{13}}}{\sum_{j}e^{x_{1j}}} & \dots  & \frac{e^{x_{1n}}}{\sum_{j}e^{x_{1j}}} \\
    \frac{e^{x_{21}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{22}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{23}}}{\sum_{j}e^{x_{2j}}} & \dots  & \frac{e^{x_{2n}}}{\sum_{j}e^{x_{2j}}} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \frac{e^{x_{m1}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m2}}}{\sum_{j}e^{x_{mj}}} & \frac{e^{x_{m3}}}{\sum_{j}e^{x_{mj}}} & \dots  & \frac{e^{x_{mn}}}{\sum_{j}e^{x_{mj}}}
\end{bmatrix} = \begin{pmatrix}
    softmax\text{(first row of x)}  \\
    softmax\text{(second row of x)} \\
    ...  \\
    softmax\text{(last row of x)} \\
\end{pmatrix} $$


```python
# 函数: softmax

def softmax(x):
    """计算输入x每一行的softmax.
    你的代码应当无论时对于一行的向量还是大小为(n, m)的矩阵都适用.
    
    参数:
    x -- 大小为(n, m)的numpy matrix
    
    返回:
    s -- 大小为(n, m),x对应项进行softmax的numpy matrix
    """
    
    ### 代码开始 ### (≈ 3 行代码)
    # 对x中每个元素求指数.使用 np.exp(...)
    x_exp = np.exp(x)
    
    # 新建一个向量x_sum , 逐行对x_exp求和. 使用 np.sum(..., axis =1, keepdims =True)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    
    # x_exp 除以 x_sum 来计算 softmax(x). numpy的广播机制自动进行.
    s = x_exp/x_sum
    
    ### 代码结束 ###
    
    return s
```


```python
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
```

    softmax(x) = [[  9.80897665e-01   8.94462891e-04   1.79657674e-02   1.21052389e-04
        1.21052389e-04]
     [  8.78679856e-01   1.18916387e-01   8.01252314e-04   8.01252314e-04
        8.01252314e-04]]
    

**期望输出**:

<table style="width:60%">

     <tr> 
       <td> **softmax(x)** </td> 
       <td> [[  9.80897665e-01   8.94462891e-04   1.79657674e-02   1.21052389e-04
    1.21052389e-04]
 [  8.78679856e-01   1.18916387e-01   8.01252314e-04   8.01252314e-04
    8.01252314e-04]]</td> 
     </tr>
</table>


**注意**:
- 假如打印 x_exp, x_sum, s的大小,你会发现x_sum的形状为 (2, 1),然而 x_exp 的形状为 (2, 5). **x_exp/x_sum** 能够运行时由于python的广播机制.

恭喜你!现在你对于Python 的numpy有了较为深入的理解,并且动手实现了一些有用的函数,他们在后面的深度学习的学习中也将用到.

<font color='blue'>
**请牢记:**
- np.exp(x) 对于任何的 np.array(x) 都使用.求出x中每个元素的指数函数值.
- sigmoid 函数以及其梯度
- image2vector 在深度学习中十分常见
- np.reshape 也很常用.在之后的学习中,你会发现,精心控制你的矩阵或向量的维度,在调试中能够减少大量的bug.
- numpy 有很多高效的内置函数.
- 广播机制非常好用.    

## 2) 向量化

在深度学习中,你会处理很大的数据集.因此在你的算法中,一个费计算最优的函数会成为算法性能的瓶颈,甚至导致一个模型需要非常长的时间才能够运行出来结果.为了保证你代码计算的高效性,你将会使用到向量化的技术.因此,你需要清楚地理解以下概念:点积(内积),外积(向量积、外积、叉积), 对应项积.


```python
import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### 常规的向量点积实现 ###
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("点积 = " + str(dot) + "\n ----- 计算耗时 = " + str(1000*(toc - tic)) + "ms")

### 常规的外积实现 ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2))) # 创建大小为 len(x1)*len(x2) 全零矩阵
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print ("外积 = " + str(outer) + "\n ----- 计算耗时 = " + str(1000*(toc - tic)) + "ms")

###  典常规的对应项相乘实现 ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("对应项乘积 = " + str(mul) + "\n ----- 计算耗时 = " + str(1000*(toc - tic)) + "ms")

### 常规的一般点积实现 ###
W = np.random.rand(3,len(x1)) # 随机的 3*len(x1) numpy array
print(W.shape)
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print ("一般点积 = " + str(gdot) + "\n ----- 计算耗时 = " + str(1000*(toc - tic)) + "ms")
```

    点积 = 278
     ----- 计算耗时 = 0.0ms
    外积 = [[ 81.  18.  18.  81.   0.  81.  18.  45.   0.   0.  81.  18.  45.   0.
        0.]
     [ 18.   4.   4.  18.   0.  18.   4.  10.   0.   0.  18.   4.  10.   0.
        0.]
     [ 45.  10.  10.  45.   0.  45.  10.  25.   0.   0.  45.  10.  25.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [ 63.  14.  14.  63.   0.  63.  14.  35.   0.   0.  63.  14.  35.   0.
        0.]
     [ 45.  10.  10.  45.   0.  45.  10.  25.   0.   0.  45.  10.  25.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [ 81.  18.  18.  81.   0.  81.  18.  45.   0.   0.  81.  18.  45.   0.
        0.]
     [ 18.   4.   4.  18.   0.  18.   4.  10.   0.   0.  18.   4.  10.   0.
        0.]
     [ 45.  10.  10.  45.   0.  45.  10.  25.   0.   0.  45.  10.  25.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]]
     ----- 计算耗时 = 0.0ms
    对应项乘积 = [ 81.   4.  10.   0.   0.  63.  10.   0.   0.   0.  81.   4.  25.   0.   0.]
     ----- 计算耗时 = 0.0ms
    (3, 15)
    一般点积 = [ 29.99637459  14.81828858  18.55331938]
     ----- 计算耗时 = 0.0ms
    


```python
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### 向量化的向量点积 ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("点积 = " + str(dot) + "\n ----- 计算耗时 = " + str(1000*(toc - tic)) + "ms")

### 向量化的外积 ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("外积 = " + str(outer) + "\n ----- 计算耗时 = " + str(1000*(toc - tic)) + "ms")

### 向量化的对应项乘积 ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("对应项乘积 = " + str(mul) + "\n ----- 计算耗时 = " + str(1000*(toc - tic)) + "ms")

### 向量化的一般点积 ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("一般的点积 = " + str(dot) + "\n ----- 计算耗时 = " + str(1000*(toc - tic)) + "ms")
```

    点积 = 278
     ----- 计算耗时 = 0.0ms
    外积 = [[81 18 18 81  0 81 18 45  0  0 81 18 45  0  0]
     [18  4  4 18  0 18  4 10  0  0 18  4 10  0  0]
     [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [63 14 14 63  0 63 14 35  0  0 63 14 35  0  0]
     [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [81 18 18 81  0 81 18 45  0  0 81 18 45  0  0]
     [18  4  4 18  0 18  4 10  0  0 18  4 10  0  0]
     [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]
     ----- 计算耗时 = 0.0ms
    对应项乘积 = [81  4 10  0  0 63 10  0  0  0 81  4 25  0  0]
     ----- 计算耗时 = 0.0ms
    一般的点积 = [ 29.99637459  14.81828858  18.55331938]
     ----- 计算耗时 = 0.0ms
    

你可能注意到了,向量化的实现更加的高效和简洁.对于更大的向量以及矩阵,向量化的实现和常规实现的运行时间差异更大.

**注意**:`np.dot()`进行的时矩阵和矩阵,或者矩阵和向量的乘积操作.和`np.multiply()`以及`*`还不一样,后者时对应项之间的乘积.

### 2.1 实现L1和L2损失函数

**练习**:实现L1的损失函数,实现方式为numpy的向量化运算版本.求绝对值函数`abs(x)`可能会用到.

**牢记**:
- 损失(loss)是用来评价你的模型的指标.你得到的损失越大,表明你的预测值($\hat {y}$)和真实值($y$)之间差异越大.在深度学习中,我们使用例如梯度下降的优化算法来训练模型和最小化cost(代价/成本).
- L1损失的定义如下:
$$
\begin{align*} & L_1(\hat{y}, y) = \sum_{i=0}^m|y^{(i)} - \hat{y}^{(i)}| \end{align*}\tag{6}
$$


```python
# 函数: L1

def L1(yhat, y):
    """
    参数:
    yhat -- 大小为m的向量 (预测标签)
    y -- 大小为m的向量 (真实标签)
    
    返回:
    loss -- L1 loss定义如上方所示
    """
    
    ### 代码开始 ###()(≈ 1 行代码)
    loss = np.sum(abs(yhat-y))
    ### 代码结束 ###
    
    return loss
```


```python
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
```

    L1 = 1.1
    

**期望输出**:

<table style="width:20%">

     <tr> 
       <td> **L1** </td> 
       <td> 1.1 </td> 
     </tr>
</table>


**练习**:实现L2 损失,方式为numpy的向量化方式.有不直一种实现方式.你可能会用到`np.dot()`.提示一点,假如$x = [x_1, x_2, ..., x_n ]$, 那么`np.dot(x, x)`= $\sum_{j=0}^n x_j^{2}$

- L2 损失定义为$$\begin{align*} & L_2(\hat{y},y) = \sum_{i=0}^m(y^{(i)} - \hat{y}^{(i)})^2 \end{align*}\tag{7}$$


```python
# 函数: L2

def L2(yhat, y):
    """
    参数:
    yhat -- 大小为m的向量 (预测标签)
    y -- 大小为m的向量 (真实标签)
    
    返回:
    loss -- L2 loss定义如上方所示
    """
    ### 代码开始 ###()(≈ 1 行代码)
    loss = np.sum((yhat-y)**2)
    ### 代码结束 ###
    
    return loss
```


```python
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
```

    L2 = 0.43
    

**期望输出**: 
<table style="width:20%">
     <tr> 
       <td> **L2** </td> 
       <td> 0.43 </td> 
     </tr>
</table>

恭喜你成功完成本次任务,我们希望通过本次的热身训练能够为你后面的任务打下一定的基础,之后的任务更加有趣和激动人心!

<font color='blue'>
**请牢记**:
- 向量化在深度学习中十分的重要.其保证了计算的高效和明晰.
- 回顾和理解L1 和 L2损失.
- 熟练使用 np.sum, np.dot, np.multiply,np.maximum的大量numpy内置函数.
