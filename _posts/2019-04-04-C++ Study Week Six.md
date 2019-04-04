---
layout:     post
title:      C++ Study Week Six
subtitle:   null
date:       2019-04-04
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

# 第六周

## 一、虚函数和多态的基本概念

### 1.1 虚函数

> - 在类的定义中，前面有virtual关键字的成员函数就是虚函数。
> - virtual关键字只用在类定义里的函数声明中，写函数体时不用。
> - 构造函数的静态成员函数不能是虚函数。

```c++
class base{
    virtual int get();
};
int base::get() { }
```

### 1.2 多态的表现形式一

> - 派生类的指针可以赋给基类指针。
> - 通过基类指针调用基类和派生类中的同名**虚函数**时：
>   - 若该指针指向一个基类的对象，那么被调用是基类的虚函数；
>   - 若该指针指向一个派生类的对象，那么被调用的是派生类的虚函数；

### 1.3 多态的表现形式二

> - 派生类的对象可以赋给基类引用。
> - 通过基类引用调用基类和派生类中的同名**虚函数**时：
>   - 若该引用引用一个基类的对象，那么被调用是基类的虚函数；
>   - 若该引用引用一个派生类的对象，那么被调用的是派生类的虚函数；

### 1.4 多态的简单示例

```c++
Class A{
    public:
    virtual void Print(){
        cout << "A::Print" << endl;
    }
};
Class B:public A{
    public:
    virtual void Print(){
        cout << "B::Print" << endl;
    }
};
Class D:public A{
    public:
    virtual void Print(){
        cout << "D::Print" << endl;
    }
};
Class E:public A{
    public:
    virtual void Print(){
        cout << "E::Print" << endl;
    }
};

int main(){
    A a;
    B b;
    D d;
    E e;
    A * pa = &a;
    B * pb = &b;
    D * pd = &d;
    E * pe = &e;
    
    pa->Print(); //a.Print()被调用，输出：A::Print
    pa = pb;
    pa->Print(); //b.Print()被调用，输出：B::Print
    pa = pd;
    pa->Print(); //d.Print()被调用，输出：D::Print
    pa = pe;
    pa->Print(); //e.Print()被调用，输出：E::Print
    return 0;
}
```

## 二、多态实例：魔法门之英雄无敌

```c++
//基类CCreature
class CCreature{
    protected:
    int m_nLifeValue,m_nPower;
    public:
    virtual void Attack(CCreature * pCreature) {}
    virtual void Hurted(int nPower) {}
    virtual void FightBack(CCreature * pCreature) {}
};
//派生类 CDragon
class CDragon{
    public:
    virtual void Attack(CCreature * pCreature) {}
    virtual void Hurted(int nPower) {}
    virtual void FightBack(CCreature * pCreature) {}
};
void CDragon::Attack(CCreature * pCreature) {
    //表现攻击动作的代码
    p->Hurted(m_nPower); //多态
    p->FightBack(this); //多态
}
void CDragon::Hurted(int nPower) {
    //表现受伤动作的代码
    m_nLifeValue -= nPower；
}
void CDragon::FightBack(CCreature * pCreature) {
     //表现反击动作的代码
    p->Hurted(m_nPower/2); //多态
}
```

## 三、多态实例：几何形体程序

 ### 3.1 根据几何形体面积排序

```c++
class CShape{
  public:
    virtual double Area() = 0; //纯虚函数
    virtual void PrintInfo() = 0;
};

class CRectangle:public CShape{
  public:
    int w, h;
    virtual double Area();
    virtual void PrintInfo();
};

class CCircle:public CShape{
  public:
    int r;
    virtual double Area();
    virtual void PrintInfo();
};

class CTriangle:public CShape{
  public:
    int a, b, c;
    virtual double Area();
    virtual void PrintInfo();
};

double CRectangle::Area(){
    return w * h;
}
void CRectangle::PrintInfo(){
    cout << "Rectangle:" << Area() << endl;
}
double CCircle::Area(){
    return 3.14 * r * r;
}
void CCircle::PrintInfo(){
    cout << "Circle:" << Area() << endl;
}
double CTriangle::Area(){
    double p = (a + b + c) / 2.0;
    return sqrt(p * (p - a) * (p - b) * (p - c));
}
void CTriangle::PrintInfo(){
    cout << "Triangle:" << Area() << endl;
}

CShape * pShapes[100];
int MyCompare(const void * s1, const void * s2){
    double a1, a2;
    CShape ** p1; //s1，s2是void*，不可写“*s1”来取得s1指向的内容
    CShape ** p2;
    p1 = (CShape **) s1; //s1，s2指向pShapes数组中的元素，数组元素的类型是CShape *
    p2 = (CShape **) s2; //故p1，p2都是指向指针的指针，类型为CShape **
    a1 = (*p1)->Area(); //*p1的类型是CShape *，是基类指针，故此句为多态
    a2 = (*p2)->Area();
    if(a1 < a2)
        return -1;
    else if(a2 < a1)
        return 1;
    else
        return 0;
}

int main(){
    int i;
    int n;
    CRectangle * pr;
    CCircle * pc;
    CTriangle * pt;
    cin >> n;
    for(i = 0;i < n;++i){
        char c;
        cin >> c;
        switch(c){
            case 'R':
                pr = new CRectangle();
                cin >> pr->w >> pr->h;
                pShapes[i] = pr;
                break;
            case 'C':
                pc = new CCircle();
                cin >> pc->r;
                pShapes[i] = pc;
                break;
            case 'T':
                pt = new CTriangle();
                cin >> pt->a >> pt->b >> pt->c;
                pShapes[i] = pt;
                break;
        }
    }
    qsort(pShapes, n, sizeof(CShape*), MyCompare);
    for(i = 0;i < n;++i){
        pShapes[i]->PrintInfo();
    }
    return 0;
}
```

### 3.2 多态的又一例子

```c++
class Base{
  public:
    void fun1() { this->fun2(); } //this是基类指针，fun2是虚函数，所以是多态
    virtual void fun2() { cout << "Base::fun2()" << endl; }
};

class Derived:public Base{
  public:
    virtual void fun2() { cout << "Derived::fun2()" << endl; }
};

int main(){
    Derived d;
    Base * pBase = & d;
    pBase->fun1(); //输出：Derived::fun2()
    return 0;
}
```

> **注：在非构造函数，非析构函数的成员函数中调用虚函数，是多态。**

### 3.3 在构造函数和析构函数中调用虚函数

```c++
class myclass{
  public:
    virtual void hello() { cout << "hello from myclass." << endl; }
    virtual void bye() { cout << "bye from myclass." << endl; }
};

class son:public myclass{
  public:
    void hello() { cout << "hello from son." << endl; } //派生类中和基类中虚函数同名同参数的函数，不加virtual也自动成为虚函数
    son() { hello(); }
    ~son() { bye(); }
};

class grandson:public son{
  public:
    void hello() { cout << "hello from grandson." << endl; }
    void bye() { cout << "bye from grandson." << endl; }
    grandson() { cout << "constructing grandson." << endl; }
    ~grandson() { cout << "destructing grandson." << endl; }
};

int main(){
    grandson gson;
    son *pson;
    pson = &gson;
    pson->hello(); //多态
    return 0;
}

//输出结果：
//hello from son.
//constructing grandson.
//hello from grandson.
//destructing grandson.
//bye from myclass.
```

> 在构造函数和析构函数中调用虚函数，不是多态。编译时即可确定，调用的函数是**自己的类或基类**中定义的函数，不会等到运行时才决定调用自己的还是派生类的函数。

## 四、多态的实现原理

> “多态”的关键在于通过基类指针或引用调用一个虚函数时，编译时不确定到底调用的是基类还是派生类的函数，运行时才确定——这叫“动态联编”。

```c++
class Base{
  public:
    int i;
    virtual void Print() { cout << "Base::Print()" << endl; }
};

class Derived:public Base{
  public:
    int n;
    virtual void Print() { cout << "Derived::Print()" << endl; }
};

int main(){
    Derived d;
    cout << sizeof(Base) << ", " << sizeof(Derived);//输出：8， 12
    return 0;
}
```

> **多态实现的关键——虚函数表**
>
> 每一个有虚函数的类（或有虚函数的类的派生类）都有一个**虚函数表**，该类的**任何对象**中都放着虚函数表的指针。虚函数表中列出了该类的虚函数地址。**多出来的4个字节就是用来存放虚函数表的地址。**如下图所示：

![class_Base](http://wx2.sinaimg.cn/large/868d8571ly1g1pkvqrrgzj20yk0h3dl4.jpg)

![class_Derived](http://wx1.sinaimg.cn/large/868d8571ly1g1pkwi23flj20za0m0dlt.jpg)

> 多态的函数调用语句被编译成一系列根据基类指针所指向的（或基类引用所引用的）对象中存放的虚函数表的地址，在虚函数表中查找虚函数地址，并调用虚函数的指令。
>
> **多态的缺点：有多余的时间和空间上的开销**。查虚函数表是时间上的开销，存指针的4个字节是空间上的开销。

```c++
class A{
  public:
    virtual void Func() { cout << "A::Func" << endl; }
};

class B:public A{
  public:
    virtual void Func() { cout << "B::Func" << endl; }
};

int main(){
    A a;
    A * pa = new B();
    pa->Func(); //输出：B::Func
    long long * p1 = (long long *) & a; //将a的地址转换为一个long long型的指针
    long long * p2 = (long long *) & pa;//将pa的地址转换为一个long long型的指针
    * p2 = * p1;
    pa->Func(); //输出：A::Func
    return 0;
}
```

## 五、虚析构函数、纯虚函数和抽象类

### 5.1 虚析构函数

> - 通过基类的指针删除派生类对象时，通常情况下只调用基类的析构函数。
>   - 但是，删除一个派生类的对象时，应该先调用派生类的析构函数，然后调用基类的析构函数。
> - 解决方法：把基类的**析构函数声明为virtual**。
>   - 派生类的析构函数的virtual可以不进行声明（但仍是虚函数）
>   - 通过基类的指针删除派生类对象时，首先调用派生类的析构函数，然后调用基类的析构函数
> - 一般来说，一个类如果定义了虚函数，则应该将析构函数也定义为虚函数。或者，一个类打算作为基类使用，也应该将析构函数定义成虚函数。
> - **注意：不允许以虚函数作为构造函数。**

```c++
class son{
  public:
    ~son() { cout << "bye from son." << endl; }
};

class grandson:public son{
  public:
    ~grandson() { cout << "bye from grandson." << endl; }
};

int main(){
    son *pson;
    pson = new grandson();
    delete pson; 
    return 0;
}
//输出：bye from son.
```

```c++
class son{
  public:
    virtual ~son() { cout << "bye from son." << endl; }
};

class grandson:public son{
  public:
    ~grandson() { cout << "bye from grandson." << endl; }
};

int main(){
    son *pson;
    pson = new grandson();
    delete pson; 
    return 0;
}
//输出：
//bye from grandson.
//bye from son. 
```

### 5.2 纯虚函数和抽象类

> **纯虚函数：没有函数体的虚函数**

```c++
class A{
  private:
    int a;
  public:
    virtual void Print() = 0; //纯虚函数
    void fun() { cout << "fun"; }
}
```

> - **包含纯虚函数的类叫抽象类。**
>
>   - 抽象类只能作为基类来派生新类使用，不能创建抽象类的对象
>
>   - 抽象类的指针和引用可以指向由抽象类派生出来的类的对象
>
>     `A a; //错，A是抽象类，不能创建对象`
>
>     `A * pa; //ok，可以定义抽象类的指针和引用`
>
>     `pa = new A; //错误，A是抽象类，不能创建对象`
>
> - **在抽象类的成员函数内可以调用纯虚函数，但是在构造函数或析构函数内部不能调用纯虚函数。**
>
> - **如果一个类从抽象类派生而来，那么当且仅当它实现了基类中的所有纯虚函数，它才能成为非抽象类。**

```c++
class A{
  public:
    virtual void f() = 0; //纯虚函数
    void g() { 
        this->f(); //ok
    } 
    A() {
        f(); //错误
    }
};

class B:public A{
  public:
    void f() { cout << "B:f()" << endl; }
};

int main(){
    B b;
    b.g();
    return 0;
}

//输出：
//B:f()
```

## 六、测验

### 6.1 Fun和Do

```c++
#include <iostream> 
using namespace std;
class A { 
	private: 
	int nVal; 
	public: 
	void Fun() 
	{ cout << "A::Fun" << endl; }; 
	void Do() 
	{ cout << "A::Do" << endl; } 
}; 
class B:public A { 
	public: 
	virtual void Do() 
	{ cout << "B::Do" << endl;} 
}; 
class C:public B { 
	public: 
	void Do( ) 
	{ cout <<"C::Do"<<endl; } 
	void Fun() 
	{ cout << "C::Fun" << endl; } 
}; 
void Call(
B & p
) { 
	p.Fun(); p.Do(); 
} 
int main() { 
	C c; 
	Call( c); 
	return 0;
}

//输出结果：
//A::Fun 
//C::Do 
```

### 6.2 怎么又是Fun和Do

```c++
class A {
	private:
	int nVal;
	public:
	void Fun()
	{ cout << "A::Fun" << endl; };
	virtual void Do()
	{ cout << "A::Do" << endl; }
};
class B:public A {
	public:
	virtual void Do()
	{ cout << "B::Do" << endl;}
};
class C:public B {
	public:
	void Do( )
	{ cout <<"C::Do"<<endl; }
	void Fun()
	{ cout << "C::Fun" << endl; }
};
void Call(
A * p
) {
	p->Fun(); p->Do();
}
int main() {
	Call( new A());
	Call( new C());
	return 0;
}

//输出结果

//A::Fun

//A::Do

//A::Fun

//C::Do
```

