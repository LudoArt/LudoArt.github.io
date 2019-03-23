---
layout:     post
title:      C++ Study Week Two
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

# 第二周

## 一、 类和对象的基本概念（2）

### 1.1 类的成员函数和类的定义分开写

```C++
// 矩形类
class CRectangle
{
    public:
    int w, h;
    int Area()； // 成员函数仅在此处声明
    int Perimeter()；
    void Init(int w_, int h_)；
}
int CRectangle::Area(){
        return w * h;
}
int CRectangle::Perimeter(){
        return 2 * (w + h);
}
void CRectangle::Init(int w_, int h_){
        w = w_; h = h_;
};
```

### 1.2 类成员的可访问范围

> - private：私有成员，只能在成员函数内访问
> - public：公有成员，可以在任何地方访问
> - protected：保护成员
> - 若没有上述关键字，则缺省地被认为是私有成员

### 1.3 成员函数的重载及参数缺省

> - 成员函数也可以重载
> - 成员函数也可以带缺省参数

```C++
class Location{
    private:
    int x, y;
    public:
    void init(int x = 0, int y = 0);
    void valueX(int val){ x = val; }
    void valueX(){ return x; }
};
```

## 二、构造函数

> **构造函数：**
>
> - 名字和类名相同，可以有参数，不能有返回值
> - 作用是对对象进行初始化，如给成员变量赋初值
> - 如果定义类时没写构造函数，则编译器生成一个默认的**无参数**的构造函数（不做任何操作）
> - 对象生成时构造函数自动调用，一旦生成，就再也不能执行构造函数
> - 一个类可以有多个构造函数

```C++
class Complex{
    private:
    double real, imag;
    public:
    Complex(double r, double i);
    Complex(double r);
    Complex(Complex c1, Complex c2);
};
Complex::Complex(double r, double i){
    real = r; imag = i;
}
Complex c1; // error, 缺少构造函数的参数
Complex * pc = new Complex; // error, 缺少构造函数的参数
Complex c1(2); // OK
Complex c1(2, 4), c2(3, 5); // OK
Complex * pc = new Complex(2, 4); // OK
Complex c3(c1, c2); // OK
```

```C++
// 构造函数在数组中的使用
class Test{
    public:
    Test(int n){} //(1)
    Test(int n, int m){} //(2)
    Test(){} //(3)
};
Test array1[3] = {1, Test(1,2)}; //三个元素分别用(1)，(2)，(3)初始化
Test array2[3] = {Test(2,3), Test(1,2), 1}; //三个元素分别用(2)，(2)，(1)初始化
Test * pArrar[3] = {new Test(4), new Test(1,2)}; //两个元素分别用(1)，(2)初始化
```

## 三、复制构造函数

### 3.1 复制构造函数的基本概念

> **复制构造函数：**
>
> - 只有一个参数，即对同类对象的引用
> - 形如 `X：：X( X& )` 或 ` X：：X( const X & )` ，二者选一
> - 若没有定义复制构造函数，将会自动生成默认的复制构造函数（完成复制功能）

```C++
// 系统自动生成默认复制构造函数
class Complex{
    private:
    double real, imag;
};
Complex c1;   // 调用缺省无参构造函数
Complex c2(c1); // 调用缺省的复制构造函数，将c2初始化成和c1一样

// 自己定义复制构造函数
class Complex{
    public:
    double real, imag;
    Complex() {}
    Complex(const Complex & c){
        real = c.real;
        imag = c.imag;
        cout << "Copy Constructor Called.";
    }
};
Complex c1;   // 调用缺省无参构造函数
Complex c2(c1); // 调用自己定义的复制构造函数，输出"Copy Constructor Called."
```

### 3.2 复制构造函数起作用的三种情况

```c++
// 1) 当用一个对象去初始化同类的另一个对象时
Complex c2(c1);
Complex c2 = c1; // 初始化语句，非赋值语句

// 2) 若某函数有一个参数是类A的对象，那么该函数被调用时，类A的复制构造函数将被调用
class A{
    public:
    A() {};
    A(A & a){
        cout << "Copy Constructor Called.";
    }
}
void Func(A a1){ }
int main(){
    A a2;
    Func(a2); // 调用了类似了A a1 = a2;这样的功能，故将调用复制构造函数，输出"Copy Constructor Called."
    return 0;
}

// 3) 如果函数的返回值是类A的对象，则函数返回时，A的复制构造函数将被调用
A Func(){
    A b;
    return b;
}
int main(){
    Func(); // 输出"Copy Constructor Called."
    return 0;
}
```

> **注：对象间的赋值并不导致复制构造函数被调用**

## 四、类型转换构造函数和析构函数

### 4.1 类型转换构造函数

> **类型转换构造函数：**
>
> - 定义转换构造函数的目的是实现类型的自动转换
> - 只有一个参数，而且不是复制构造函数的构造函数，一般就可以看作是转换构造函数
> - 当需要的时候，编译系统会自动调用转换构造函数，建立一个无名的临时对象（或临时变量）

```c++
class Complex{
    public:
    double real, imag;
    Complex(int i) {//类型转换构造函数
        cout << "IntConstructor Called." << endl;
        real = i; imag = 0;
    }
    Complex(double r, double i){
        real = r; imag = i;
    }
};
int main()
{
    Complex c1(7, 8);
    Complex c2 = 12;
    c1 = 9; // 将9自动转换成一个临时的Complex对象
    cout << c1.real << ", " << c1.imag << endl; // 输出 9, 0
}
```

### 4.2 析构函数

> 析构函数：
>
> - 名字与类名相同，在前面加 “~” ，没有参数和返回值，**一个类最多只能有一个析构函数**
> - 析构函数在对象消亡时自动被调用
> - 如果定义类时没写析构函数，则编译器生成缺省析构函数，缺省析构函数什么都不做

```c++
class Srting{
    private:
    char * p;
    public:
    String(){
        p = new char[10];
    }
    //析构函数
    ~ String(){
        delete []p;
    }
}
int main(){
    String *s = new String[3]; //构造函数调用3次
    delete [] s; //析构函数调用3次
}
```

## 五、构造函数析构函数调用时机

```c++
class Demo{
    int id;
    public:
    Demo(int i){
        id = i;
        cout << "id = " << id << " constructed." << endl;
    }
    ~Demo(){
        cout << "id = " << id << " destructed." << endl;
    }
}

Demo ld(1);
void Func(){
    static Demo d2(2);
    Demo d3(3);
    cout << "func" << endl;
}
int main(){
    Demo d4(4);
    d4 = 6;
    cout << "main" << endl;
    { Demo d5(5); }
    Func();
    cout << "main ends" << endl;
    return 0;
}

// 输出结果：
// id = 1 constructed.
// id = 4 constructed.
// id = 6 constructed.
// id = 6 destructed.
// main
// id = 5 constructed.
// id = 5 destructed.
// id = 2 constructed.
// id = 3 constructed.
// func
// id = 3 destructed.
// main ends
// id = 6 destructed.
// id = 2 destructed.
// id = 1 destructed.
```

## 六、测验

###  6.1 奇怪的类复制

```c++
class Sample {
public:
	int v;
	Sample() {
		v = 0;
	}
	Sample(int i) {
		v = i;
	}
	Sample(const Sample & x) {
		v = x.v + 2;
	}
};
void PrintAndDouble(Sample o)
{
	cout << o.v;
	cout << endl;
}
int main()
{
	Sample a(5); //调用Sample(int i)
	Sample b = a; //调用Sample(const Sample & x)
	PrintAndDouble(b); //传入参数需进行复制，故调用Sample(const Sample & x)，输出9
	Sample c = 20; //调用Sample(int i)
	PrintAndDouble(c); //传入参数需进行复制，故调用Sample(const Sample & x)，输出22
	Sample d; //调用Sample()
	d = a; //不调用复制构造函数，故d.v = a.v
	cout << d.v; //输出5
	return 0;
}

//输出结果：
//9
//22
//5
```

