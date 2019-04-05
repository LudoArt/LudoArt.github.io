---
layout:     post
title:      C++ Study Week One
subtitle:   null
date:       2019-03-23
author:     LudoArt
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - C++

---

<head>
	<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
	<script src="https://cdn.rawgit.com/google/code-prettify/master/loader/run_prettify.js">
	</script>
</head>

# 第一周

## 一、 引用

### 1.1  引用的概念

```C++
int n =4;
int & r = n; // r引用了n，r的类型是int &
```

*某个变量的引用，等价于这个变量，相当于该变量的一个别名*

```C++
int n = 7;
int & r = n;
r = 4;
cout << r;  // 4

cout << n;  // 4

n = 5;
cout << r;  // 5

```

> 注：
>
> - 定义引用时一定要将其**初始化**成引用某个变量。
> - 初始化后，它就一直引用该变量，不会再引用别的变量。
> - 引用只能引用**变量**，不能引用常量和表达式。

```c++
int a = 7, b = 8;
int & r1 = a;
int & r2 = r1; // r2也引用a

r2 = 10;
cout << a;  // 10

r1 = b;     // r1没有引用b

cout << a;  // 8

```

### 1.2  引用的应用

#### 1.2.1 引用作为函数的参数

``` C++
void swap(int & a, int & b)
{
    int tmp;
    tmp = a; a = b; b = tmp;
}
```

#### 1.2.2 引用作为函数的返回值

```C++
int n = 4;
int & SetValue() { return n; }
int main()
{
    SetValue() = 40;
    cout << n; // 40
    
    return 0;
}
```

### 1.3 常引用

```C++
int n = 100;
const int & r = n;
r = 200; //编译出错

n = 300; //没问题

```

## 二、const关键字

### 2.1 定义常量

```C++
const int MAX_VAL = 23;
```

### 2.2 定义常量指针

```C++
// 不可以通过常量指针修改其指向的内容

int n,m;
const int * p = & n;
* p = 5; // 编译出错

n = 4; // ok

p = & m; // ok，常量指针的指向可以改变

```

```C++
// 不能把常量指针赋值给非常量指针，反之可以

const int * p1;
int * p2;
p2 = p1; // 编译出错

p2 = (int *) p1; // ok

p1 = p2; // ok

```

## 三、动态内存分配

> new 和 delete，较熟悉就不记了，注意释放数组时使用 delete [ ] a 即可

## 四、内联函数和重载函数

### 4.1  内联函数

``` C++
// 内联函数：在函数定义前添加关键字“inline”

// 优点：减少函数调用的开销

// 缺点：程序体积增大

inline int Max(int a, int b)
{
    if(a > b) return a;
    return b;
}
```

### 4.2 重载函数

```C++
// 函数重载：一个或多个函数，名字相同，参数个数或类型不同 

// 若名字相同，参数个数和类型也相同，返回值类型不同，称之为函数的重复定义

int Max(double f1, double f2){}
int Max(int n1, int n2){}
int Max(int n1, int n2, int n3){}
```

### 4.3 函数的缺省参数

```C++
void func(int x1, int x2 = 2, int x3 = 3) { }
func(10); // ok

func(10, 8); // ok

func(10, ,8); // error

```

## 五、类和对象的基本概念

> 面向对象的程序设计 = 类 + 类 + 类 + ……
>
> 类 = 属性（成员变量） + 行为（成员函数）

```C++
// 矩形类

class CRectangle
{
    public:
    // 矩形的属性（宽和高）
    
    int w, h;
    // 矩形的行为（求面积、求周长、初始化）
    
    int Area(){
        return w * h;
    }
    int Perimeter(){
        return 2 * (w + h);
    }
    void Init(int w_, int h_){
        w = w_; h = h_;
    }
}
```

> 通过类，可以定义变量，类定义出来的变量，也成为类的实例，就是“对象”

```C++
int main()
{
    int w,h;
    CRectangle r; // r是一个对象
    
    cin >> w >> h;
    r.Init(w, h);
    cout << r.Area() << endl;
    return 0;
}
```

## 六、测验

### 6.1 难一点的swap

```C++
void swap(int *& a, int *& b) // 对int*类型的变量进行引用
    
{
	int * tmp = a;
	a = b;
	b = tmp;
}
int main()
{
	int a = 3,b = 5;
	int * pa = & a;
	int * pb = & b;
	swap(pa,pb);
	cout << *pa << "," << * pb;
	return 0;
}
```

### 6.2 神秘的数组初始化

```C++
int main()
{
	int * a[] = {NULL,NULL,new int,new int[6]}; 
	
	*a[2] = 123;
	a[3][5] = 456;
	if(! a[0] ) {
		cout << * a[2] << "," << a[3][5];
	}
	return 0;
}
```

