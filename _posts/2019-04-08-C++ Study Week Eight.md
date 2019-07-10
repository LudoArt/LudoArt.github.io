---
layout:     post
title:      C++ Study Week Eight
subtitle:   null
date:       2019-04-08
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

# 第八周

## 一、string类

### 1.1 string类

```c++
int main()
{
	string s1("Hello");
	cout << s1 << endl; //Hello

	string s2(8, 'x');
	cout << s2 << endl; //xxxxxxxx
    
	string month = "March";
	cout << month << endl; //March
    
	string s;
	s = 'n';
	cout << s << endl; //n
    
    string error('n'); //error
    
	return 0;
}
```

> - string对象的长度用成员函数 `length()` 读取；
>   -  `strng s("hello");`
>   -  `cout << s.length() << endl;`
> - string支持流读取运算符
>   -  `string stringObj;` 
>   -  `cin >> stringObj;` 
> - string支持 `getline` 函数
>   -  `string s;`
>   -  `getline(cin, s);` 

### 1.2 string的赋值和连接

> - 用 `=` 赋值
>   -  `string s1("cat"), s2;` 
>   -  `s2 = s1;` 
> - 用 `assign` 成员函数复制
>   -  `string s1("cat"), s3;` 
>   -  `s3.assign(s1);` 
> - 用 `assign` 成员函数部分复制
>   -  `string s1("catpig"), s3;` 
>   -  `s3.assign(s1, 1, 3); //从s1中下标为1的字符开始复制3个字符给s3`
> - 单个字符复制
>   -  `s2[5] = s1[3] = 'a';`  
> - 逐个访问string对象中的字符
>   -  `string s1("Hello");` 
>   -  `for(int i = 0;i < s1.length(); ++i) cout << s1.at(i) << endl;` 
> - 成员函数at会做**范围检查**，如果超出范围，会抛出 `out_of_range` 异常，而**下标运算符[]**不做范围检查。
> - 用 `+` 运算符连接字符串
>   -  `string s1("good"), s2("morning!");` 
>   -  `s1 += s2;``
> - 用成员函数 `append` 连接字符串
>   -  `string s1("good"), s2("morning!");` `
>   -  `s1.append(s2);` `
>   -  `s2.append(s1, 3, s1.size()); //s1.size(), s1字符数，下标为3开始，s1.size()个字符，如果字符串内没有足够字符，则复制到字符串最后一个字符` 

### 1.3 比较string

> - 用关系运算符比较string的大小
>   -  ==, >, >=, <, <=, !=
>   - 返回值都是bool类型，成立返回true，否则返回false
> - 用成员函数 `compare` 比较string的大小

```c++
string s1("hello"), s2("hello"), s2("hell");
int f1 = s1.compare(s2); //0 (hello == hello)

int f2 = s1.compare(s3); //1 (hello > hell)

int f3 = s3.compare(s1); //-1 (hell < hello)

int f4 = s1.compare(1, 2, s3, 0, 3); //-1 (el < hell)

int f5 = s1.compare(0, s1.size(), s3); //1 (hello > hell)
```

### 1.4 子串

> - 成员函数 `substr`

```c++
string s1("hello world"), s2;
s2 = s1.substr(4,5); //下标4开始的5个字符

cout << s2 << endl; //o wor
```

### 1.5 交换string

> - 成员函数 `swap`

```c++
string s1("hello world"), s2("really");
s1.swap(s2); 
cout << s1 << endl; //really

cout << s2 << endl; //hello world
```

###  1.6 寻找string中的字符

> - 成员函数 `find(string s)` ：从前向后查找，返回s第一次出现的位置的下标，查找失败返回 `strin::npos` 
> - 成员函数 `rfind(string s)` ：从后向前查找，返回s第一次出现的位置的下标，查找失败返回 `strin::npos` 
> - 成员函数 `find_first_of(string s)` ：从前向后查找，返回s中任意一个字符第一次出现的位置的下标，查找失败返回 `strin::npos` 
> - 成员函数 `find_last_of(string s)` ：从后向前查找，返回s中任意一个字符第一次出现的位置的下标，查找失败返回 `strin::npos` 
> - 成员函数 `find_first_not_of(string s)` ：从前向后查找，返回不在s中的任意一个字符第一次出现的位置的下标，查找失败返回 `strin::npos` 
> - 成员函数 `find_last_not_of(string s)` ：从后向前查找，返回不在s中的任意一个字符第一次出现的位置的下标，查找失败返回 `strin::npos` 

```c++
string s1("hello world");
cout << s1.find("ll") << endl; //2

cout << s1.find("abc") << endl; //4294967295(strin::npos)

cout << s1.rfind("ll") << endl; //9

cout << s1.rfind("abc") << endl; //4294967295(strin::npos)

cout << s1.find_first_of("abcde") << endl; //1

cout << s1.find_first_of("abc") << endl; //4294967295(strin::npos)

cout << s1.find_last_of("abcde") << endl; //11

cout << s1.find_last_of("abc") << endl; //4294967295(strin::npos)

cout << s1.find_first_not_of("abcde") << endl; //0

cout << s1.find_first_not_of("hello world") << endl; //4294967295(strin::npos)

cout << s1.find_last_not_of("abcde") << endl; //10

cout << s1.find_last_not_of("hello world") << endl; //4294967295(strin::npos)
```

### 1.7 删除string中的字符

> - 成员函数 `erase()`
> - 成员函数 `replace()`

```c++
string s1("hello world");
s1.erase(5); //去掉下标5及之后的字符

cout << s1 << " " << s1.length() << " " << s1.size() << endl; //hello 5 5

s1 = "hello world";
s1.replace(2, 3, "haha"); //将s1中下标2开始的3个字符换成“haha”

cout << s1 << endl; //hehaha world

s1 = "hello world";
s1.replace(2, 3, "haha", 1, 2); //将s1中下标2开始的3个字符换成“haha”中下标1开始的2个字符

cout << s1 << endl; //heah world
```

### 1.8 在string中插入字符

> - 成员函数 `insert()`

```c++
string s1("hello world");
string s2("show insert");
s1.insert(5, s2); //将s2插入到s1下标5的位置

cout << s1 << endl; //helloshow insert world

s1.insert(2, s2, 5, 3); //将s2中下标5开始的3个字符插入s1下标2的位置

cout << s1 << endl; //heinslloshow insert world
```

### 1.9 转换成C语言式的char*字符串

> - 成员函数 `c_str()`
> - 成员函数 `data()`

```c++
string s1("hello world");
printf("%s\n", s1.c_str()); //s1.c_str()返回传统的const char *类型字符串，且该字符串以 '\0' 结尾

//输出：hello world

string s1("hello world");
const char *p1 = s1.data();
for(int i = 0;i < s1.length(); ++i)
    printf("%c", *(p1 + i)); //s1.data()返回一个char *类型字符串，对s1的修改可能会使p1出错

//输出：hello world
```

### 1.10字符串拷贝

> - 成员函数 `copy()`

```c++
string s1("hello world");
int len = s1.length();
char * p2 = new char[len + 1];
s1.copy(p2, 5, 0); //从s1的下标0的字符开始，制作一个最长5个字符长度的字符串副本，并将其赋值给p2。返回值表明实际复制的字符串的长度

p2[5] = 0;
cout << p2 << endl; //hello
```

### 1.11 字符串流处理

> - 除了标准流和文件流输入输出外，还可以从string进行输入输出；
> - 类似 `istream` 和 `ostream` 进行标准流输入输出，我们用 `istringstream` 和 `ostringstream` 进行字符串上的输入输出，也成为内存输入输出。

```c++
#include <string>
#include <iostream>
#include <sstream>

int main()
{
    //istringstream示例
    
	string input("Input test 123 4.7 A");
	istringstream inputString(input);
	string string1, string2;
	int i;
	double d;
	char c;
	inputString >> string1 >> string2 >> i >> d >> c;
	cout << string1 << endl; //Input
    
	cout << string2 << endl; //test
    
	cout << i << endl; //123 
    
	cout << d << endl; //4.7 
    
	cout << c << endl; //A
    
	long L;
	if (inputString >> L)
		cout << "long\n";
	else
		cout << "empty\n"; //empty
    
    //ostringstream示例
    
    ostringstream outputString;
	int a = 10;
	outputString << "This" << a << "ok" << endl;
	cout << outputString.str(); //This 10ok
    
	return 0;
}
```

## 二、标准模板库STL概述（一）

### 2.1 容器概述

> - **顺序容器**
>   -  `vector, deque, list`
> - **关联容器**
>   -  `set, multiset, map, multimap`
> - **容器适配器**
>   -  `stack, queue, priority_queue`

### 2.2 顺序容器简介

> - **vector**——动态数组
> - **deque**——双向队列
> - **list**——双向链表

### 2.3 关联容器简介

> - 元素是**排序**的
> - 插入任何元素，都按其相应的排序规则来确定其位置
> - 在查找时具有非常好的性能
> - 通常以平衡二叉树方式实现，**插入和检索的时间都是O(longN)**
> - **set/multiset**：集合，set中不允许相同元素，multiset中允许存在相同的元素
> - **map/multimap**：map中存放的元素有且仅有两个成员变量，一个名为first，另一个名为second，map**根据first值对元素进行从小到大排序**，并可快速地根据first来检索元素。map中不允许相同first值的元素，multimap中允许存在相同first值的元素

### 2.4 容器适配器简介

> - **stack**——栈
> - **queue**——队列
> - **priority_queue**——优先级队列

### 2.5 顺序容器和关联容器中都有的成员函数

> - **begin**：返回指向容器中第一个元素的迭代器
> - **end**：返回指向容器中最后一个元素后面的位置的迭代器
> - **rbegin**：返回指向容器中最后一个元素的迭代器
> - **rend**：返回指向容器中第一个元素签名的位置的迭代器
> - **erase**：从容器中删除一个或几个元素
> - **clear**：从容器中删除所有元素

### 2.6 顺序容器的常用成员函数

> - **front**：返回容器中第一个元素的引用
> - **back**：返回容器中最后一个元素的引用
> - **push_back**：在容器末尾中增加新元素
> - **pop_back**：删除容器末尾中的新元素
> - **erase**：删除迭代器指向的元素（可能会使该迭代器失效），或删除一个区间，返回被删除元素后面的那个元素的迭代器

### 2.7 迭代器

> - 用于指向顺序容器和关联容器中的元素
> - 迭代器用法和指针类似
> - 有const和非const两种
> - 通过迭代器可以读取它指向的元素
> - 通过非const迭代器还能修改其指向的元素
> - 定义一个容器类迭代器的方法：
>   - 容器类名::iterator 变量名;
>   - 容器类名::const_iterator 变量名;
> - 访问一个迭代器指向的元素：
>   - *迭代器变量名

### 2.8 双向迭代器

> **若p和p1都是双向迭代器，则**
>
> - **++p, p++**：使p指向容器中下一个元素
> - **--p, p--**：使p指向容器中上一个元素
> - ***p**：取p指向的元素
> - **p  = p1**：赋值
> - **p == p1, p != p1**：判断是否相等、不等

### 2.9 随机访问迭代器

> **若p和p1都是随机访问迭代器，则**
>
> - 双向迭代器的所有操作
> - **p += i**：使p向后移动i个元素
> - **p -= i**：使p向前移动i个元素
> - **p + i**：值为：指向p后面的第i个元素的迭代器
> - **p - i**：值为：指向p前面的第i个元素的迭代器
> - **p[i]**：值为：p后面的第i个元素的引用
> - **p < p1, p <= p1, p > p1, p >= p1**
> - **p - p1**：p和p1之间的元素个数

|     容器     | 容器上的迭代器类别 |
| :----------: | :----------------: |
|    vector    | 随机访问 |
|    deque     | 随机访问 |
|     list     | 双向 |
| set/multiset | 双向 |
| map/multimap |  双向   |
|    stack     |      不支持迭代器      |
|    queue     |      不支持迭代器      |
|    priority_queue     |      不支持迭代器      |

### 2.10 STL中“大”“小”的概念

> - **关联容器**内部的元素是**从小到大**排序的
> - 有些算法要求其操作的区间是**从小到大**排序的，称为“**有序区间算法**”，例如：`binary_search`
> - 有些算法会对区间进行**从小到大**排序，称为“**排序算法**”，例如：`sort`
> - 还有一些其他算法会用到“大”，“小”的概念
> - **使用STL时，在缺省的情况下，以下三个说法等价：**
>   1. x 比 y 小
>   2. 表达式 “x < y” 为真
>   3. y 比 x 大

### 2.11 STL中“相等”的概念

> - 有时，“x和y相等”等价于“**x==y为真**”，例如：在未排序的区间上进行的算法，如顺序查找 `find`
> - 有时，“x和y相等”等价于“**x小于y和y小于x同时为假**”，例如：有序区间算法，如`binary_search`；关联容器自身的成员函数 `find`

```c++
#include <iostream>
#include <algorithm>
using namespace std;

class A {
	int v;
public:
	A(int n) :v(n) { }
	bool operator < (const A & a2) const {
		//必须为常量成员函数
        
		cout << v << "<" << a2.v << "?" << endl;
		return false;
	}
	bool operator == (const A & a2) const {
		//必须为常量成员函数
        
		cout << v << "==" << a2.v << "?" << endl;
		return v == a2.v;
	}
};

int main()
{
	A a[] = { A(1),A(2) ,A(3) ,A(4) ,A(5) };
	cout << binary_search(a, a + 4, A(9)); //折半查找
    
	return 0;
}

//输出：

//3<9?

//2<9?

//1<9?

//9<1?

//1
```

## 三、vector，deque和list

> vector、deque略

### 3.3 list 容器

> - 在任何位置插入删除都是常数时间，不支持随机存取
> - 除了具有所有顺序容器都有的成员函数以外，还支持8个成员函数：
>   - **push_front**：在前面插入元素
>   - **pop_front**：删除前面的元素
>   - **sort**：排序
>   - **remove**：删除和指定值相等的所有元素
>   - **unique**：删除所有和**前一个元素相同**的元素（要做到元素不重复，则unique之前还需要sort）
>   - **merge**：合并两个链表，并清空被合并的那个
>   - **reverse**：颠倒链表
>   - **splice**：在指定位置前面插入另一个链表中的一个或多个元素，并在另一链表中删除被插入的元素

## 四、函数对象

### 4.1 函数对象

> 是个对象，但是用起来看上去像函数调用，实际上也执行了函数调用

```c++
class CMyAverage {
public:
	double operator()(int a1, int a2, int a3) {
		//重载()运算符
        
		return (double)(a1 + a2 + a3) / 3;
	}
};

int main()
{
	CMyAverage average; //函数对象
    
	cout << average(3, 2, 3); // average.operator()(3, 2, 3) 用起来看上起像函数调用
    
	//输出 2.6667
    
	return 0;
}
```

### 4.2 函数对象的应用

STL里有以下模板：

```c++
template<class InIt, class T, class Pred>
T accumulate(InIt first, InIt last, T val, Pred pr);
```

其中：

- pr 就是个函数对象
  - 对 [first, last) 中的每个迭代器 I ，执行 val = pr(val, * I)，返回最终的val
- pr 也可以是个函数

```c++
//Dev C++ 中 Accumulate 源代码1：

template<typename _InputIterator, typename _Tp>
_Tp accumulate(_InputIterator _first, _InputIterator _last, _Tp _init) {
	for (; _first!= _last; ++_first)
		_init = _init + *_first;
	return _init;
}

//Dev C++ 中 Accumulate 源代码2：

template<typename _InputIterator, typename _Tp, typename _BinaryOperation>
_Tp accumulate(_InputIterator _first, _InputIterator _last, _Tp _init, _BinaryOperation _binary_op) {
	for (; _first!= _last; ++_first)
		_init = _binary_op(_init, *_first);
	return _init;
}
```

```c++
int sumSquares(int total, int value) {
	return total + value * value;
}

template<class T>
void PrintInterval(T first, T last) {
	//输出区间[first, last)中的元素
    
	for (; first != last; ++first)
		cout << *first << " ";
	cout << endl;
}

template<class T>
class SumPowers {
private:
	int power;
public:
	SumPowers(int p) :power(p) {}
	const T operator()(const T & total, const T & value) {
		//计算value的power次方，加到total上
        
		T v = value;
		for (int i = 0; i < power - 1; ++i)
			v = v * value;
		return total + v;
	}
};

int main()
{
	const int SIZE = 10;
	int a1[] = { 1,2,3,4,5,6,7,8,9,10 };
	vector<int> v(a1, a1 + SIZE);
	cout << "1) ";
	PrintInterval(v.begin(), v.end());
	int result = accumulate(v.begin(), v.end(), 0, sumSquares);
	cout << "2) 平方和：" << result << endl;
	result = accumulate(v.begin(), v.end(), 0, SumPowers<int>(3));
	cout << "3) 立方和：" << result << endl;
	result = accumulate(v.begin(), v.end(), 0, SumPowers<int>(4));
	cout << "4) 4次方和：" << result << endl;
	return 0;
}

//输出：

//1) 1 2 3 4 5 6 7 8 9 10

//2) 平方和：385

//3) 立方和：3025

//4) 4次方和：25333
```

```c++
//其中accumulate实例化出：
int accumulate(vector<int>::iterator first, vector<int>::iterator last, int init, int (*op)(int, int)){
    for(;first != last;++first)
        init = op(init, *first);
    return init;
}
```

## 五、测验

