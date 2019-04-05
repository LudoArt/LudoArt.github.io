---
layout:     post
title:      C++ Study Week Five
subtitle:   null
date:       2019-04-02
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

# 第五周

## 一、继承和派生的基本概念

> - 派生类拥有基类的全部成员函数和成员变量，不论是private、protected、public。
>
> - 在派生类的各个成员函数中，不能访问基类中的private成员

```c++
class CStudent{
  private:
    string sName;
    int nAge;
  public:
    bool IsThreeGood() {};
    void SetName(const string & name)
    {
        sName = name;
    }
};

//派生类的写法是：类名:public 基类名

class CUndergraduateStudent:public CStudnet{
  private:
    int nDepartment;
  public:
    bool IsThreeGood() { ... }; //覆盖
    
    bool CanBaoYan() { ... };
}

class CGraduateStudent:public CStudnet{
  private:
    int nDepartment;
    char szMentorName[20];
  public:
    bool CountSalary() { ... }; 
}
```

> 派生类对象的体积，等于基类对象的体积，再加上派生类对象自己的成员变量的体积。
>
> 在派生类对象中，包含在基类对象，而且基类对象的存储位置位于派生类对象新增的成员变量之前

```c++
class CStudent{
  private:
    string name;
    string id;//学号
    
    char gender//性别
        
    int age;
  public:
    void PrintInfo();
    void SetInfo(const string & name_, const string & id_, int age_, char gender_);
    void GetName() { return name; }
}

//本科生类，继承了CStudnet类

class CUndergraduateStudent:public CStudnet{
  private:
    string department;//学生所属的系的名称
    
  public:
    //给予保研资格
    
    void QualifiedForBaoyan(){
        cout << "qualified for baoyan" << endl;
    }
    void PrintInfo(){
        CStudnet::PrintInfo();//调用基类的PrintInfo
        
        cout << "Department:" << department << endl;
    }
    void SetInfo(const string & name_, const string & id_, int age_, char gender_, const string & department_){
        CStudnet::SetInfo(name_, id_, age_, gender_);//调用基类的SetInfo
        
        department = department_;
    }
    void GetName() { return name; }
}
```

## 二、继承关系和复合关系

> - **继承**： “是” 关系。
>   - 基类A，B是基类A的派生类。
>   - 逻辑上要求：“一个B对象也是一个A对象” 。
> - **复合**： “有” 关系
>   - 类C中 “有” 成员变量k，k是类D的对象，则C和D是复合关系。
>   - 逻辑上要求： “D对象是C对象的固有属性或组成部分” 。

```c++
//复合关系的使用

//正确的写法：

//  为 “狗” 类设一个 “业主” 类的对象指针

//  为 “业主” 类设一个 “狗” 类的对象指针数组

class CMaster;//CMaster必须提前声明，不能先写CMaster类后写CDog类

class CDog{
    CMaster * pm;
};
class CMaster{
    CDog * dogs[10];
}
```

## 三、覆盖和保护成员 

### 3.1 覆盖

> 派生类可以定义一个和基类成员同名的成员，这叫**覆盖**。在派生类中访问这类成员时，缺省的情况是访问派生类中定义的成员。要在派生类中访问由基类定义的**同名成员**时，要使用**作用域符号::**。

```c++
class base{
    int j;
  public:
    int i;
    void func();
};
class derived:public base{
  public:
    int i; //覆盖
    
    void access();
    void func(); //覆盖
    
}
void derived::access(){
    j = 5; //error
    
    i = 5; //引用的是派生类的i
    
    base::i = 5; //引用的是基类的i
    
    func(); //派生类的
    
    base::func(); //基类的
    
}

derived obj;
obj.i = 1; //引用的是派生类的i

obj.base::i = 1; //引用的是基类的i

```

> **一般来说，基类和派生类不定义同名成员变量。**

### 3.2 保护成员

- 基类的 `private` 成员：可以被下列函数访问
  - 基类的成员函数
  - 基类的友元函数
- 基类的 `public` 成员：可以被下列函数访问
  - 基类的成员函数
  - 基类的友元函数
  - 派生类的成员函数
  - 派生类的友元函数
  - 其他的函数
- 基类的 `protected` 成员：可以被下列函数访问
  - 基类的成员函数
  - 基类的友元函数
  - 派生类的成员函数可以访问**当前对象的基类**的保护成员

```c++
class Father{
    private: int nPrivate; //私有成员
    
    public: int nPublic; //公有成员
    
    protected: int nProtected; //保护成员
    
};
class Son: public Father{
    void AccessFather(){
        nPublic = 1; //ok
        
        nPrivate = 1; //wrong
        
        nProtected = 1; //ok，访问从基类继承的protected成员
        
        Son f；
            f.nProtected = 1; //wrong，f不是当期对象
        
    }
}
int main(){
    Father f;
    Son s;
    f.nPublic = 1; //ok
    
    s.nPublic = 1; //ok
    
    f.nProtected = 1; //error
    
    s.nProtected = 1; //error
    
    f.nPrivate = 1; //error
    
    s.nPrivate = 1; //error
    
    return 0;
}
```

## 四、派生类的构造函数

```c++
class Bug{
  private:
    int nLegs;
    int nColor;
  public:
    int nType;
    Bug(int legs, int color);
    void PrintBug() {};
};
//派生类

class FlyBug:public Bug{
    int nWings;
  public:
    FlyBug(int legs, int color, int wings);
};
Bug::Bug(int legs, int color){
    nLegs = legs;
    nColor = color;
}
//错误的FlyBug构造函数

FlyBug::FlyBug(int legs, int color, int wings){
    nLegs = legs; //不能访问
    
    nColor = color; //不能访问
    
    nType = 1; //ok
    
    nWings = wings;
}
//正确的FlyBug构造函数

FlyBug::FlyBug(int legs, int color, int wings):Bug(legs, color){
    nWings = wings;
}
```

> - 在创建派生类的对象时，需要调用基类的构造函数：初始化派生类对象中从基类继承的成员。在执行一个派生类的构造函数之前，总是先执行基类的构造函数。
> - 调用基类构造函数的两种方式：
>   - 显示方式：在派生类的构造函数中，为基类的构造函数提供参数。
>   - 隐式方式：在派生类的构造函数中，省略基类构造函数时，派生类的构造函数则自动调用基类的默认构造函数（无参构造函数，若无，则编译报错）。
> - 派生类的析构函数被执行时，执行完派生类的析构函数后，自动调用基类的析构函数。

```c++
class Base{
  public:
    int n;
    Base(int i):n(i){
        cout << "Base" << n << " constructed" << endl;
    }
    ~Base(){
        cout << "Base" << n << " destructed" << endl;
    }
};
class Derived:public Base{
  public:
    Derived(int i):Base(i){
        cout << "Derived constructed" << endl;
    }
    ~Derived(){
        cout << "Derived destructed" << endl;
    }
};
int main(){
    Derived Obj(3);
    return 0;
}

//输出结果：

//Base 3 constructed

//Derived constructed

//Derived destructed

//Base 3 destructed
```

> **封闭派生类对象的构造函数执行顺序**
>
> - 在创建派生类的对象时：
>
>   1. 先执行基类的构造函数，用以初始化派生类对象中从基类继承的成员；
>   2. 再执行成员对象类的构造函数，用以初始化派生类对象中成员对象；
>   3. 最后执行派生类自己的构造函数；
> - 在派生类对象消亡时：
>
>   1. 先执行派生类自己的析构函数；
>   2. 再依次执行各成员对象类的析构函数；
>   3. 最后执行基类的析构函数。
>   
> - 析构函数的调用顺序与构造函数的调用顺序相反。

## 五、公有继承的赋值兼容规则

### 5.1 public继承的赋值兼容规则

```c++
class Base{ };
class Derived:public Base{ };
Base b;
Derived d;
```
> 1. 派生类的对象可以赋值给基类对象
> `b = d;`
>
> 2. 派生类对象可以初始化基类引用
> `base & br = d;`
>
> 3. 派生类对象的地址可以赋值给基类指针
> `base * pb = & d;`
>
> **如果派生方式是private或protected，则上述三条不可行。**

### 5.2 直接基类与间接基类

> - 类A派生类B，类B派生类C，类C派生类D，……
>   - 类A是类B的直接基类
>   - 类B是类C的直接基类，类A是类C的间接基类
>   - 类C是类D的直接基类，类A、B是类D的间接基类
> - 在声明派生类时，只需要列出它的直接基类，派生类沿着类的层次自动向上继承它的间接基类
> 	- 派生类的成员包括
> 	  1. 派生类自己定义的成员
> 	  2. 直接基类中的所有成员
> 	  3. 所有间接基类的全部成员

## 六、测验 

### 6.1 全面的MyString 通过码

```c++
int strlen(const char * s) 
{	int i = 0;
	for(; s[i]; ++i);
	return i;
}
void strcpy(char * d,const char * s)
{
	int i = 0;
	for( i = 0; s[i]; ++i)
		d[i] = s[i];
	d[i] = 0;
		
}
int strcmp(const char * s1,const char * s2)
{
	for(int i = 0; s1[i] && s2[i] ; ++i) {
		if( s1[i] < s2[i] )
			return -1;
		else if( s1[i] > s2[i])
			return 1;
	}
	return 0;
}
void strcat(char * d,const char * s)
{
	int len = strlen(d);
	strcpy(d+len,s);
}
class MyString
{
private:
    char *pstr;
public:
    MyString()
    {
        pstr = new char[1];
        *pstr = '\0';
    }
    MyString(const char *str)
    {
        int len = strlen(str);
        pstr = new char[len + 1];
        strcpy(pstr,str);
    }
    MyString(const MyString &rhs)
    {   
        int len = strlen(rhs.pstr);
        pstr = new char[len + 1];
        strcpy(pstr,rhs.pstr);
    }
    MyString & operator=(const MyString & rhs)
    {
        if(this == & rhs)
            return *this;
        delete []pstr;
        int len = strlen(rhs.pstr);
        pstr = new char[len + 1];
        strcpy(pstr,rhs.pstr);
        return *this;
    }
    MyString & operator +=(const MyString &rhs)
    {
        MyString temp(*this);
        return *this = temp + rhs;
    }
    friend MyString operator+(const MyString & x,const MyString & y)
    {
        char * temp = new char[strlen(x.pstr) + strlen(y.pstr) + 1];
        strcpy(temp,x.pstr);
        strcat(temp,y.pstr);
        MyString re(temp);
        delete []temp;
        return re ;
    }
    friend ostream & operator <<(ostream & os,const MyString & rhs)
    {
        os << rhs.pstr;
        return os;
    }
    friend bool operator<(const MyString &x,const MyString & y)
    {
        return strcmp(x.pstr,y.pstr) < 0;
    }
    friend bool operator==(const MyString &x,const MyString & y)
    {
        return strcmp(x.pstr,y.pstr) == 0;
    }
    friend bool operator>(const MyString &x,const MyString & y)
    {
        return strcmp(x.pstr,y.pstr) > 0;
    }
    char& operator[](int n)
    {
        return *(pstr + n);
    }
    MyString operator()(int b,int len)
    {
        char * temp = new char[strlen(pstr)+1];
        strcpy(temp,pstr+b);
        *(temp+len) = '\0';
        MyString re(temp);
        delete []temp;
        return re;
    }
    ~MyString(){delete[]pstr;}

};


int CompareString( const void * e1, const void * e2)
{
	MyString * s1 = (MyString * ) e1;
	MyString * s2 = (MyString * ) e2;
	if( * s1 < *s2 )
	return -1;
	else if( *s1 == *s2)
	return 0;
	else if( *s1 > *s2 )
	return 1;
}
int main()
{
	MyString s1("abcd-"),s2,s3("efgh-"),s4(s1);
	MyString SArray[4] = {"big","me","about","take"};
	cout << "1. " << s1 << s2 << s3<< s4<< endl;
	s4 = s3;
	s3 = s1 + s3;
	cout << "2. " << s1 << endl;
	cout << "3. " << s2 << endl;
	cout << "4. " << s3 << endl;
	cout << "5. " << s4 << endl;
	cout << "6. " << s1[2] << endl;
	s2 = s1;
	s1 = "ijkl-";
	s1[2] = 'A' ;
	cout << "7. " << s2 << endl;
	cout << "8. " << s1 << endl;
	s1 += "mnop";
	cout << "9. " << s1 << endl;
	s4 = "qrst-" + s2;
	cout << "10. " << s4 << endl;
	s1 = s2 + s4 + " uvw " + "xyz";
	cout << "11. " << s1 << endl;
	qsort(SArray,4,sizeof(MyString),CompareString);
	for( int i = 0;i < 4;i ++ )
	cout << SArray[i] << endl;
	//s1的从下标0开始长度为4的子串
    
	cout << s1(0,4) << endl;
	//s1的从下标5开始长度为10的子串
    
	cout << s1(5,10) << endl;
	return 0;
}
```

### 6.2 继承自string的MyString 通过码

```c++
class MyString:public string
{
public:
	MyString():string() {}
	MyString(const char *s):string(s) {}
	MyString(const MyString & s):string(s){}
	MyString(const string & s):string(s) {}
	MyString operator()(int start, int end)
	{
		return substr(start, end);
	}

};

int main()
{
	MyString s1("abcd-"),s2,s3("efgh-"),s4(s1);
	MyString SArray[4] = {"big","me","about","take"};
	cout << "1. " << s1 << s2 << s3<< s4<< endl;
	s4 = s3;
	s3 = s1 + s3;
	cout << "2. " << s1 << endl;
	cout << "3. " << s2 << endl;
	cout << "4. " << s3 << endl;
	cout << "5. " << s4 << endl;
	cout << "6. " << s1[2] << endl;
	s2 = s1;
	s1 = "ijkl-";
	s1[2] = 'A' ;
	cout << "7. " << s2 << endl;
	cout << "8. " << s1 << endl;
	s1 += "mnop";
	cout << "9. " << s1 << endl;
	s4 = "qrst-" + s2;
	cout << "10. " << s4 << endl;
	s1 = s2 + s4 + " uvw " + "xyz";
	cout << "11. " << s1 << endl;
        sort(SArray,SArray+4);
	for( int i = 0;i < 4;i ++ )
	cout << SArray[i] << endl;
	//s1的从下标0开始长度为4的子串
    
	cout << s1(0,4) << endl;
	//s1的从下标5开始长度为10的子串
    
	cout << s1(5,10) << endl;
	return 0;
}
```

