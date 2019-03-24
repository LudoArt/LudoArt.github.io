---
layout:     post
title:      C++ Study Week Three
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

# 第三周

## 一、this指针

> **非静态**成员函数中可以直接使用this来代表指向该函数作用的对象的指针。

```c++
class A{
    int i;
    public:
    void Hello(){ cout << "hello" << endl; }
};

int main(){
    A * p = NULL;
    p -> Hello();
}
//输出结果：hello
```

可以这么理解，编译器将 `void Hello(){ cout << "hello" << endl; } ` 翻译成C语言的形式，即  `void Hello(A *this){ cout << "hello" << endl; }` ，那么 `p -> Hello();` 也便翻译为 `Hello(p);` ，函数 `Hello()` 中，输出语句与this指针所指的对象无关，故可以直接输出hello。

```c++
class A{
    int i;
    public:
    void Hello(){ cout << i << "hello" << endl; }
};

int main(){
    A * p = NULL;
    p -> Hello(); // error
}
```

分析同上，编译器将 `void Hello(){ cout << i <<  "hello" << endl; } ` 翻译成C语言的形式，即  `void Hello(A *this){ cout << this -> i <<  "hello" << endl; }` ，因为此时this指针为空，故会报错。

> **注：静态成员函数中不能使用this指针，因为静态成员函数并不具体作用于某个对象。**

## 二、静态成员变量

> **静态成员：在说明前面加了 `static` 关键字的成员，不需要通过对象就能访问。**
>
> - 普通成员变量：每个对象有各自的一份。
> - 静态成员变量：只有一份，为所有对象共享，本质上是全局变量。
> - `sizeof` 运算符不会计算静态成员变量。
> - 普通成员函数：必须具体作用于某个对象。
> - 静态成员函数：并不具体作用于某个对象，本质上是全局函数。

```c++
class CRectangle
{
    private:
    int w, h;
    static int nTotalArea; //静态成员变量
    static int nTotalNumber;
    public:
    CRectangle(int w_, int h_);
    ~CRectangle();
    static void PrintTotal(); //静态成员函数
};
CRectangle::CRectangle(int w_, int h_){
    w = w_;
    h = h_;
    nTotalNumber++;
    nTotalArea += w * h;
}
CRectangle::~CRectangle(){
    nTotalNumber--;
    nTotalArea -= w * h;
}
void CRectangle::PrintTotal(){
    cout << nTotalNumber << "," << nTotalArea << endl;
}

//必须在定义类的文件中对静态成员变量进行一次说明或初始化。否则编译能通过，链接不能通过
int CRectangle::nTotalNumber = 0;
int CRectangle::nTotalArea = 0;

int main(){
    CRectangle r1(3, 3), r2(2, 2);
    cout << CRectangle::nTotalNumber; // error, nTotalNumber为私有变量
    CRectangle::PrintTotal(); //2,13
    r1.PrintTotal(); //2,13
    return 0;
}

//访问静态成员
// 1) 类名::成员名
CRectangle::PrintTotal();
// 2) 对象名.成员名
CRectangle r; r.PrintTotal();
// 3) 指针->成员名
CRectangle *p = &r; p -> PrintTotal();
// 4) 引用.成员名
CRectangle & ref = r; int n = ref.nTotalNumber;
```

> **注：在静态成员函数中，不能访问非静态成员变量，也不能调用非静态成员函数。**

```c++
//在使用CRectangle类时，有时会调用复制构造函数生成临时的隐藏的CRectangle对象，此时nTotalArea和nTotalNumber的值就会出错（临时对象生成时没有增加nTotalArea和nTotalNumber，但是在消亡时会调用析构函数减少nTotalArea和nTotalNumber）
//解决方法：为CRectangle类写一个复制构造函数
CRectangle::CRectangle(CRectangle & r){
    w = r.w;
    h = r.h;
    nTotalNumber++;
    nTotalArea += w * h;
}
```

## 三、成员对象和封闭类

有成员对象的类叫封闭类

```c++
//轮胎类
class CTyre{
    private:
    int radius;
    int width;
    public:
    CTyre(int r, int w):radius(r), width(w) {}
};
//引擎类
class CEngine{
    
};
//汽车类（封闭类）
class CCar{
    private:
    int price;
    CTyre tyre;
    CEngine engine;
    public:
    CCar(int p, int tr, int tw):price(p), tyre(tr, tw) {}
};

//若CCar类不定义构造函数，则CCar car; 会编译出错
//因为编译器不知道car.tyre该如何初始化
//car.engine的初始化没问题
```

> 封闭类构造函数和析构函数的执行顺序：
>
> - 封闭类对象生成时，先执行所有对象成员的构造函数，然后才执行封闭类的构造函数
> - 对象成员的构造函数调用次序和对象成员在类中的说明次序一致，与它们在成员初始化列表中出现的次序无关
> - 当封闭类的对象消亡时，先执行封闭类的析构函数，然后再执行成员对象的析构函数（次序和构造函数的调用次序相反）

```c++
//封闭类的复制构造函数
class A{
    public:
    A() { cout << "default" << endl; }
    A(A & a) { cout << "copy" << endl; }
};
class B { A a; };
int main(){
    B b1, b2(b1);
    return 0;
}

//输出结果：
//default
//copy
//说明b2.a是用类A的复制构造函数初始化的
//调用复制构造函数时的实参就是b1.a
```

## 四、常量对象、常量成员函数

> - 常量对象：在定义该对象时在前面加 `const` 关键字
> - 常量成员函数：在类的成员函数说明后面加 `const` 关键字，在常量成员函数中不能修改成员变量的值，也不能调用同类的非常量成员函数（静态成员变量和静态成员函数除外）

```c++
class Sample{
    public:
    int value;
    void GetValue() const; //常量对象
    void func() {};
    Sample() {}
};
void Sample::GetValue() const{//常量函数
    value = 0; //error
    func(); //error
}

int main(){
    const Sample o;
    o.value = 100; //error 常量对象不可被修改
    o.func(); //error 常量对象上不能执行非常量成员函数
    o.GetValue(); //ok
    return 0;
}
```

> 两个成员函数，名字和参数表都一样，但是一个是 `const` ，一个不是，算重载。

```c++
class CTest{
    private:
    int n;
    public:
    CTest() { n = 1; }
    int GetValue() const { return n; }
    int GetValue() { return 2 * n; }
};
int main(){
    const CTest obj1;
    CTest obj2;
    cout << obj1.GetValue() << "," << obj2.GetValue(); // 1,2
}
```

## 五、友元

> 友元分为友元函数和友元类
>
> - 友元函数：一个类的友元函数可以访问该类的私有成员
> - 友元类：如果A是B的友元类，那么A的成员函数可以访问B的私有成员
> - 友元类之间的关系不能传递，不能继承

```c++
//友元函数举例
class CCar{
  private:
    int price;
    friend int MostExpensiveCar(CCar cars[], int total); //声明友元函数
    friend void CDriver::ModifyCar(CCar * pCar); //声明友元函数
};

class CDriver{
  public:
    //改装汽车
    void ModifyCar(CCar * pCar){
        pCar -> price += 1000; //汽车改装后价值增加，访问了CCar类中的私有成员
    }
};

int MostExpensiveCar(CCar cars[], int total){
    int tmpMax = -1;
    for(int i = 0;i < total; ++i){
        if(cars[i].price > tmpMax) //访问了CCar类中的私有成员
            tmpMax = cars[i].price; 
    }
    return tmpMax;
}
```

```c++
//友元类举例
class CCar{
  private:
    int price;
    friend class CDriver； //声明友元类
};

class CDriver{
  public:
    CCar myCar;
    //改装汽车
    void ModifyCar(){
        myCar.price += 1000; //汽车改装后价值增加，访问了CCar类中的私有成员
    }
};
```

## 六、测验

### 6.1 统计动物数量

```c++
//本题考察“静态成员变量”该知识点，但继承和虚析构函数的含义尚不明朗
//由题可知，似乎在子类对象生成的时候，也会调用父类对象的构造函数
//若Animal类不定义成虚析构函数，最后delete c2;不能正确的将Cat::number减一
class Animal {
public:
	static int number;
	Animal() {
		number++;
	}
	virtual ~Animal() {
		number--;
	}
};

class Dog :public Animal {
public:
	static int number;
	Dog() {
		number++;
	}
	~Dog() {
		number--;
	}
};

class Cat :public Animal {
public:
	static int number;
	Cat() {
		number++;
	}
	~Cat() {
		number--;
	}
};

int Animal::number = 0;
int Dog::number = 0;
int Cat::number = 0;

void print() {
	cout << Animal::number << " animals in the zoo, " << Dog::number << " of them are dogs, " << Cat::number << " of them are cats" << endl;
}

int main() {
	print(); //0 animals in the zoo, 0 of them are dogs, 0 of them are cats
	Dog d1, d2;
	Cat c1;
	print(); //3 animals in the zoo, 2 of them are dogs, 1 of them are cats
	Dog* d3 = new Dog();
	Animal* c2 = new Cat;
	Cat* c3 = new Cat;
	print(); //6 animals in the zoo, 3 of them are dogs, 3 of them are cats
	delete c3;
	delete c2;
	delete d3;
	print(); //3 animals in the zoo, 2 of them are dogs, 1 of them are cats
}
```

### 6.2 这个指针哪来的

```c++
struct A
{
	int v;
	A(int vv) :v(vv) { }
    //第一个const代表返回一个常量指针，第二个const代表这是一个常量函数
	const A * getPointer() const {
		return this;
	}
};

int main()
{
	const A a(10);
    //因为A是const类型的，所以函数getPointer()须是常量函数，否则无法调用
    //又因为p是const A *类型的，所以函数getPointer()的返回值也必须是const A *类型
	const A * p = a.getPointer();
	cout << p->v << endl;
	return 0;
}
```

### 6.3  返回什么才好呢

```c++
class A {
public:
	int val;
	A(int k = 123) {
		val = k;
	}
    //返回当前对象的一个引用
	A & GetObj() {
		return *this;
	}
};
int main()
{
	int m,n;
	A a;
	cout << a.val << endl;
	while(cin >> m >> n) {
        //将函数写在左边，变量写在右边，考虑函数的返回值是引用
        //a.GetObj()返回对象a的引用，即a = m；，调用a的类型转换，将m自动转换成一个临时的A对象
        a.GetObj() = m;
        cout << a.val << endl;
        a.GetObj() = A(n);
        cout << a.val<< endl;
	}
	return 0;
}
```



