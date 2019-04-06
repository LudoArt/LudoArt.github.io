---
layout:     post
title:      C++ Study Week Seven
subtitle:   null
date:       2019-04-07
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

# 第七周

## 一、输入输出流相关的类

### 1.1 与输入输出流操作相关的类

-  `isteam` 是用于输入的流类， `cin` 就是该类的对象。
-  `osteam` 是用于输入的流类， `cout` 就是该类的对象。
-  `ifstream` 是用于从文件读取数据的类。
-  `ofstream` 是用于向文件写入数据的类。
-  `iostream` 是既能用于输入，又能用于输出的类。
-  `fstream` 是既能从文件读取数据，又能向文件写入数据的类。

### 1.2 标准流对象

- 输入流对象： `cin` 与标准输入设备相连
- 输出流对象： 
  - `cout` 与标准输出设备相连
  -  `cerr` 与标准错误输出设备相连
  -  `clog` 与标准错误输出设备相连
-  `cin` 对应于标准输入流，用于从键盘读取数据，也可以被重定向为从文件中读取数据。
-  `cout` 对应于标准输出流，用于向屏幕输出数据，也可以被重定向为向文件写入数据。 
-  `cerr` 对应于标准错误输出流，用于向屏幕输出出错信息。 
-  `clog` 对应于标准错误输出流，用于向屏幕输出出错信息。 
-  `cerr` 和 `clog` 的区别在于 `cerr` 不使用缓冲区，直接向显示器输出信息；而输出到 `clog` 中的信息先会被存放在缓冲区，缓冲区满或者刷新时才会输出到屏幕。

### 1.3 输出重定向

```c++
int main(){
    int x, y;
    cin >> x >> y;
    freopen("test.txt", "w", stdout); //将标准输出重定向到test.txt文件
    
    if(y == 0) //除数为0则在屏幕上输出错误信息
        
        cerr << "error." << endl;
    else
        cout << x / y; //输出结果到test.txt
    
    return 0;
}
```

### 1.4 输入重定向

```c++
int main(){
    double f;
    int n;
    freopen("t.txt", "r", stdin); //cin被改为从t.txt中读取数据
    
    cin >> f >> n;
    cout << f << ", " << n << endl;
    return 0;
}
//t.txt

//3.14 123

//输出

//3.14, 123
```

### 1.5 判断输入流结束

- 如果是从文件输入，比如 `freopen("some.txt", "r", stdin);` ，那么，读到文件尾部，输入流就算结束；
- 如果从键盘输入，则在单独一行输入 Ctrl + Z  代表输入流结束。

### 1.6  `istream` 类的成员函数

1.  `istream & getline(char *buf, int bufSize);` 

从输入流中读取 `bufSize - 1` 个字符到缓冲区 `buf` ，或读到碰到 ‘\n’ 为止。

2.  `istream & getline(char *buf, int bufSize, char delim);`

从输入流中读取 `bufSize - 1` 个字符到缓冲区 `buf` ，或读到碰到 `delim` 字符为止。

3. 可以用 `if(!cin.getline(…))` 判断输入是否结束。
4.  `boof eof();` 判断输入流是否结束
5.  `int peek();` 返回下一个字符，但不从流中去掉
6.  `istream & putback(char c);` 将字符c放回输入流
7.  `istream & ignore(int nCount = 1, int delim = EOF);` 从流中删掉最多 `nCount` 个字符，遇到 `EOF` 时结束

## 二、用流操纵算子控制输出格式

### 2.1 整数流的基数：流操纵算子dec，oct，hex，setbase

```c++
int n = 10;
cout << n << endl; //10  

cout << hex << n << endl; //a(16进制)  

cout << dec << n << endl; //10(10进制)  

cout << oct << n << endl; //12(8进制)  

//此类的流操纵算子是长效的  
```

### 2.2 浮点数的精度：precision，setprecision

-  `precision` 是成员函数，其调用方式为： `cout.precision(5);` 
-  `setprecision` 是流操纵算子，其调用方式为： `cout << setprecision(5); //可以连续输出` 
- 它们功能相同。
  - **指定输出浮点数的有效位数（非定点方式输出时）**
  - **指定输出浮点数的小数点后的有效位数（定点方式输出时，定点方式：小数点必须出现在个位数后面）**

```c++
#include <iomanip>
int main(){
    double x = 1234567.89;
    double y = 12.34567;
    int n = 1234567;
    int m = 12;
    cout << setprecision(6) << x << endl << y < endl << n << endl << m;
}
//非定点方式输出：  

//1.23457e+006  

//12.3457  

//1234567  

//12  
```

```c++
#include <iomanip>
int main(){
    double x = 1234567.89;
    double y = 12.34567;
    int n = 1234567;
    int m = 12;
    cout << setiosflags(ios::fixed) << setprecision(6) << x << endl << y << endl << n << endl << m;
}
//定点方式输出：  

//123457.890000  

//12.345670  

//1234567  

//12  
```

### 2.3 设置域宽：setw，width

-  `setw` 是流操纵算子，其调用方式为： `cin >> setw(4);` 或者 `cout << setw(4);` 
-  `width` 是成员函数，其调用方式为： `cin.width(4);` 或者 `cout.width(4);` 
- 宽度设置有效性是一次性的，在每次读入和输出之前都要设置宽度

```c++
int w = 4;
char string[10];
cin.width(5);
while(cin >> string){
    cout.width(w++);
    cout << string << endl;
    cin.width(5);
}
//输入：  

//1234567890  

//输出：  

//1234(域宽为4)   

// 5678(域宽为5)  

//    90(域宽为6)  
```

### 2.4 流操纵算子的综合示例

```c++
#include <iomanip>
int main(){
    int n = 141;
    //1) 分别以十六进制、十进制、八进制先后输出n  
    
    cout << "1)" << hex << n << " " << dec << n << " " << oct << n << endl;
    //1)8d 141 215  
    
    double x = 1234567.89;
    double y = 12.34567;
    //2) 保留5位有效数字  
    
    cout << "2)" << setprecision(5) << x << " " << y << endl;
    //2)1.2346e+006 12.346   
    
    //3) 保留小数点后面5位  
    
    cout << "3)" << fixed << setprecision(5) << x << " " << y << endl;
    //3)1234567.89000 12.34567  
    
    //4) 科学计数法输出，且保留小数点后面5位   
    
    cout << "4)" << scientific << setprecision(5) << x << " " << y << endl;
    //4)1.23457e+006 1.23457e+001  
    
    //5) 非负数要显示正号，输出宽度为12字符，宽度不足则用`*`填充   
    
    cout << "5)" << showpos << fixed << setw(12) << setfill('*') << 12.1 << endl;
    //5)***+12.10000   
    
    //6) 非负数不显示正号，输出宽度为12字符，宽度不足则右边用填充字符填充   
    
    cout << "6)" << noshowpos << setw(12) << left << 12.1 << endl;
    //6)12.10000****   
    
    //7) 输出宽度为12字符，宽度不足则左边用填充字符填充   
    
    cout << "7)" << setw(12) << right << 12.1 << endl;
    //7)****12.10000   
    
    //8) 宽度不足时，负号和数值分列左右，中间用填充字符填充   
    
    cout << "8)" << setw(12) << internal << -12.1 << endl;
    //8)-***12.10000   
    
    cout << "9)" << 12.1 << endl;
    //9)12.10000  
    
}
```


### 2.5 用户自定义的流操纵算子

```c++
ostream &tab(ostream &output){
    return output << '\t';
}
cout << "aa" << tab << "bb" << endl;
//输出：aa	  bb
```

## 三、文件读写（一）

### 3.1 创建文件

* `#include <fstream>//包含头文件`

* `ofstream outFile("clients.dat", ios::out|ios::binary);//创建文件`

  * `clients.dat` 	要创建的文件的名字

  * `ios::out` 	文件打开方式
    *  `ios::out` 	输出到文件，删除原有内容
    *  `ios::app` 	输出到文件，保留原有内容，总是在尾部添加

  * `ios:binary` 	以二进制文件格式打开文件

* 也可以先创建 `ofstream` 对象，再用 `open` 函数打开

```c++
ofstream fout;
fout.open("clients.dat", ios::out|ios::binary);
```

* 判断打开是否成功
```c++
if(!fout){
    cout << "File open error!" << endl;
}
```

* 文件名可以给出绝对路径，也可以给相对路径。没有交代路径信息，就是在当前文件夹下查找文件。

### 3.2 文件的绝对路径和相对路径

- 绝对路径：

  `"c:\\tmp\\mydir\\some.txt"`

- 相对路径：
  `"\\tmp\\mydir\\some.txt"`
    当前盘符的根目录下的 *tmp\mydir\some.txt*
  `"tmp\\mydir\\some.txt"`
    当前文件夹的*tmp*子文件夹里面的...
  `"..\\tmp\\mydir\\some.txt"`
    当前文件夹的父文件夹下面的*tmp*子文件夹里面的...
  `"..\\..\\tmp\\mydir\\some.txt"`
    当前文件夹的父文件夹的父文件夹下面的*tmp*子文件夹里面的...

### 3.3 文件的读写指针

> - 对于输入文件，有一个读指针
> - 对于输出文件，有一个写指针
> - 对于输入输出文件，有一个读写指针
> - 标识文件操作的当前位置，该指针在哪里，读写操作就在哪里进行

```c++
ofstream fout("a1.out", ios::app); //以添加方式打开

long location = fout.tellp(); //取得写指针的位置

location = 10; //location可以为负值

fout.seekp(location); //将写指针移动到第10个字节处

fout.seekp(location, ios::beg); //从头部数location

fout.seekp(location, ios::cur); //从当前位置数location

fout.seekp(location, ios::end); //从尾部数location
```

```c++
ofstream fin("a1.out", ios::ate); //打开文件，定位文件指针到文件尾

long location = fin.tellp(); //取得读指针的位置

location = 10; //location可以为负值

fin.seekg(location); //将读指针移动到第10个字节处

fin.seekg(location, ios::beg); //从头部数location

fin.seekg(location, ios::cur); //从当前位置数location

fin.seekg(location, ios::end); //从尾部数location
```

### 字符文件读写实例

> 写一个程序，将文件in.txt里面的整数排序后，输出到out.txt中
>
> 例：
>
> in.txt的内容为：1 234 9 45 6 879
>
> out.txt的内容为：1 6 9 45 234 879

```c++
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;
int main(){
    vector<int> v;
    ifstream srcFile("in.txt", ios::in);
    ofstream destFile("out.txt", ios::out);
    int x;
    while(srcFile >> x)
        v.push_back(x);
    sort(v.begin(),v.end());
    for(int i = 0;i < v.size();i++)
        destFile << v[i] << " ";
    destFile.close();
    srcFile.close();
    return 0;
}
```

## 四、文件读写（二）

### 4.1 二进制读文件

`ifstream` 和 `fstream` 的成员函数： `istream& read(char* s, long n);`

将文件读指针指向的地方的n个字节内容，读入到内存地址s，然后将文件读指针向后移动n字节（以 `ios::in` 方式打开文件时，文件读指针开始指向文件开头）。 

### 4.2 二进制写文件

`ofstream` 和 `fstream` 的成员函数： `ostream& write(const char* s, long n);`

将内存地址s的n个字节内容，写入到文件写指针指向的地方，然后将文件写指针向后移动n字节（以 `ios::out` 方式打开文件时，文件读指针开始指向文件开头，以 `ios::app` 方式打开文件时，文件读指针开始指向文件尾部）。 

### 4.3 二进制文件读写

> 在文件中写入和读取一个整数

```c++
#include <iostream>
#include <fstream>
using namespace std;
int main() {
	ofstream fout("some.dat", ios::out | ios::binary);
	int x = 65;
	fout.write((const char *)(&x), sizeof(int)); //此时some.dat文件用记事本打开，显示“A”（即65对应的ASCII码）
    
	fout.close();
	ifstream fin("some.dat", ios::in | ios::binary);
	int y;
	fin.read((char *)(&y), sizeof(int));
	fin.close();
	cout << y << endl; //65
    
	return 0;
}
```

> 从键盘输入几个学生的姓名和成绩，并以二进制文件形式保存

```c++
#include <iostream>
#include <fstream>
using namespace std;

struct Student
{
	char name[20];
	int score;
};

int main() {
	Student s;
	ofstream fout("some.dat", ios::out | ios::binary);
	while (cin >> s.name >> s.score)
		fout.write((const char *)(&s), sizeof(s));
	fout.close(); //此时some.dat中的内容为“Tom 烫烫烫烫烫烫烫烫<   Jack 烫烫烫烫烫烫烫蘌   Jane 烫烫烫烫烫烫烫?   ”
    
	return 0;
}
```

> 将some.dat文件的内容读出并显示

```c++
#include <iostream>
#include <fstream>
using namespace std;

struct Student
{
	char name[20];
	int score;
};

int main() {
	Student s;
	ifstream fin("some.dat", ios::in | ios::binary);
	if (!fin) {
		cout << "error" << endl;
		return 0;
	}
	while (fin.read((char*)(&s),sizeof(s)))
	{
		int readedBytes = fin.gcount(); //刚才读了多少字节
        
		cout << s.name << " " << s.score << endl;
	}
	fin.close();
	return 0;
}
//输出：

//Tom 60

//Jack 80

//Jane 40
```

> 将some.dat文件的Jane的名字改成Mike

```c++
#include <iostream>
#include <fstream>
using namespace std;

struct Student
{
	char name[20];
	int score;
};

int main() {
	Student s;
	fstream iofile("some.dat", ios::in | ios::out | ios::binary);
	if (!iofile) {
		cout << "error" << endl;
		return 0;
	}
	iofile.seekp(2 * sizeof(s), ios::beg); //定位写指针到第三个记录
    
	iofile.write("Mike", strlen("Mike") + 1);
	iofile.seekg(0, ios::beg); //定位读指针到开头
    
	while (iofile.read((char*)(&s),sizeof(s)))
		cout << s.name << " " << s.score << endl;
	iofile.close();
	return 0;
}
//输出：

//Tom 60

//Jack 80

//Mike 40
```

> 文件拷贝，输入“mycopy src.dat dest.dat”，将src.dat拷贝到dest.dat中，若dest.dat原来就存在，则会被覆盖

```c++
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char *argv[]) {
	if (argc != 3) {
		cout << "File name missing!" << endl;
		return 0;
	}
	ifstream infile(argv[1], ios::in | ios::binary); //打开文件用于读
    
	if (!infile) {
		cout << "Source file open error." << endl;
		return 0;
	}
	ofstream outfile(argv[2], ios::out | ios::binary); //打开文件用于写
    
	if (!outfile) {
		cout << "New file open error." << endl;
		infile.close(); //打开的文件一定要关闭
        
		return 0;
	}

	char c;
	while (infile.get(c)) //每次读取一个字符
        
		outfile.put(c); //每次写入一个字符
    
	outfile.close();
	infile.close();
	return 0;
}
```

> **二进制文件和文本文件的区别**
>
> - Linux，Unix下的换行符号：'\n' 
> - Windows下的换行符号：'\r\n' 	`endl` 就是‘\n’
> - Mac OS下的换行符号：'\r' 
> - Linux/Unix下打开文件，用不用 `ios::binary` 没区别
> - Windows下打开文件，如果不用 `ios::binary` ，则：
>   - 读取文件时，所有的 ‘\r\n’ 会被当做一个字符 '\n' 处理，即少读了一个字符‘ \r’ 
>   - 写入文件时，写入单独的 ‘\n’ 是，系统自动在前面加一个 ‘\r’ ，即多写了一个 ‘\r’ 

## 五、函数模板

### 5.1 函数模板的定义

> template <class 类型参数1， class 类型参数2， ……>
> 返回值类型 模板名（形参表）
> {
> 	函数体
> };

### 5.2 函数模板中可以有不止一个类型参数

```c++
template <class T1, class T2>
T2 print(T1 arg1, T2 arg2){
	cout << arg1 << " " << arg2 << endl;
	return arg2;
}
```

### 5.3 不通过参数实例化函数模板

```c++
template <class T>
T Inc(T n){
    return 1 + n;
}
int main(){
    cout << Inc<double>(4)/2; //输出：2.5
    
    return 0;
}
```

### 5.4 函数模板的重载

> 函数模板可以重载，只要它们的形参表或类型参数表不同即可。

```c++
template <class T1, class T2>
void print(T1 arg1, T2 arg2){
	cout << arg1 << " " << arg2 << endl;
}

template <class T>
void print(T arg1, T arg2){
	cout << arg1 << " " << arg2 << endl;
}

template <class T, class T2>
void print(T arg1, T arg2){
	cout << arg1 << " " << arg2 << endl;
}
```

### 5.5 函数模板和函数的次序

> 在有多个函数和函数模板名字相同的情况下，编译器如下处理一条函数调用语句
>
> 1. 先找参数完全匹配的**普通函数**（非由模板实例化而得的函数）
> 2. 再找参数完全匹配的**模板函数**
> 3. 再找实参经过自动类型转换后能够匹配的**普通函数**
> 4. 上面都找不到，则报错

```c++
template <class T>
void Max(T arg1, T arg2){
	cout << "TemplateMax" << endl;
}

template <class T, class T2>
void Max(T arg1, T2 arg2){
	cout << "TemplateMax2" << endl;
}

void Max(double a, double b){
	cout << "MyMax" << endl;
}

int main(){
	int i = 4, j = 5;
	Max(1.2, 3.4); //输出MyMax
	
	Max(i, j); //输出TemplateMax
	
	Max(1.2, 3); //输出TemplateMax2
	
	return 0;
}
```

> **注：匹配模板函数时，不进行类型自动转换**

### 5.6 函数模板示例：Map

```c++
template<class T,class Pred>
void Map(T s, T e, T x, Pred op) {
	for (; s != e; ++s, ++x) {
		*x = op(*s);
	}
}

int Cube(int x) {
	return x * x * x;
}

double Square(double x) {
	return x * x;
}

int a[5] = { 1,2,3,4,5 }, b[5];
double d[5] = { 1.1,2.1,3.1,4.1,5.1 }, c[5];

int main() {
	Map(a, a + 5, b, Square);
	for (int i = 0; i < 5; ++i)
		cout << b[i] << ","; //1,4,9,16,25,
    
	cout << endl;

	Map(a, a + 5, b, Cube);
	for (int i = 0; i < 5; ++i)
		cout << b[i] << ","; //1,8,27,64,125,
    
	cout << endl;

	Map(d, d + 5, c, Square);
	for (int i = 0; i < 5; ++i)
		cout << c[i] << ","; //1.21,4.41,9.61,16.81,26.01,
    
	cout << endl;
	return 0;
}
```

```c++
Map(a, a + 5, b, Square); //实例化出以下函数：

void Map(int* s, int* e, int* x, double (*op)(double)) {
	for (; s != e; ++s, ++x) {
		*x = op(*s);
	}
}
```

## 六、类模版

### 6.1 类模板的定义

> template <class 类型参数1， class 类型参数2， ……> //类型参数表
> class类模板名
> {
> 	成员函数和成员变量
> };
>
> **类模板里成员函数的写法：**
> template <class 类型参数1， class 类型参数2， ……> //类型参数表
> 返回值类型 类模板名<类型参数名列表>::成员函数名(参数表)
> {
> 	……
> }
>
> **用类模板定义对象的写法：**
> 类模板名<真实类型参数表> 对象名(构造函数实参表);

### 6.2 类模板示例：Pair类模板

```c++
template<class T1,class T2>
class Pair {
public:
	T1 key;
	T2 value;
	Pair(T1 k, T2 v) :key(k), value(v) {};
	bool operator < (const Pair<T1, T2> & p) const;
};

template<class T1, class T2>
bool Pair<T1, T2>::operator < (const Pair<T1, T2> & p) const {
	return key < p.key;
}

int main() {
	//实例化出一个类Pair<string, int> 
	
	Pair<string, int> student("Tom", 19);
	cout << student.key << " " << student.value << endl;
	return 0;
}
```

### 6.3 函数模板作为类模板成员

```c++
template<class T>
class A {
public:
	template<class T2>
    //成员函数模板
        
	void Func(T2 t) {
		cout << t;
	}
};

int main() {
	A<int> a;
	a.Func('K'); //成员函数模板Func被实例化
    
	a.Func("hello"); //成员函数模板Func再次被实例化
    
	return 0;
}
//输出：Khello
```

### 6.4 非模板与非类型参数

> **类模板的“<类型参数表>”中可以出现非类型参数**

```c++
template<class T, int size>
class CArray {
	T array[size];
public:
	void Print() {
		for (int i = 0; i < size; ++i)
			cout << array[i] << endl;
	}
};

CArray<double, 40> a2;
CArray<int, 50> a3;
//a2和a3属于不同的类
```

## 七、类模版与派生、友元和静态成员变量

### 7.1 类模板与继承 

> - 类模板从类模板派生
> - 类模板从模板类派生
> - 类模板从普通类派生
> - 普通类从模板类派生

#### 7.1.1 类模板从类模板派生

```c++
template<class T1, class T2>
class A{
	T1 v1;
	T2 v2;
};

template<class T1, class T2>
class B:public A<T2, T1>{
	T1 v3;
	T2 v4;
};

template<class T>
class C:public B<T, T>{
	T v5;
};

int main(){
    B<int, double> obj1;
    C<int> obj2;
    return 0;
}
```

#### 7.1.2 类模板从模板类派生

```c++
template<class T1, class T2>
class A{
	T1 v1;
	T2 v2;
};

template<class T>
class B:public A<int, double>{
	T v;
};

int main(){
    B<char> obj1; //自动生成两个模板类：A<int, double> 和 B<char>
    
    return 0;
}
```

#### 7.1.3 类模板从普通类派生

```c++
class A{
	int v1;
};

template<class T>
class B:public A{ //所有从B实例化得到的类，都以A为基类
    
	T v;
};

int main(){
    B<char> obj1; 
    return 0;
}
```

#### 7.1.4 普通类从模板类派生

```c++
template<class T>
class A{
	T v1;
    int n;
};


class B:A<int>{
    double v;
};

int main(){
    B obj1; 
    return 0;
}
```

### 7.2 类模板与友元

> - 函数、类、类的成员函数作为类模板的友元
> - 函数模板作为类模板的友元
> - 函数模板作为类的友元
> - 类模板作为类模板的友元

#### 7.2.1 函数、类、类的成员函数作为类模板的友元

```c++
void Func1() { }
class A { };
class B{
  public:
    void Func() { }
};
template<class T>
class Tmp1{
    friend void Func1();
    friend class A;
    friend void B::Func();
};//任何从Tmp1实例化来的类，都有以上三个友元
```

#### 7.2.2 函数模板作为类模板的友元

```c++
template<class T1,class T2>
class Pair {
public:
	T1 key;
	T2 value;
	Pair(T1 k, T2 v) :key(k), value(v) {};
	bool operator < (const Pair<T1, T2> & p) const;
	template<class T3,class T4>
	friend ostream & operator<<(ostream & o, const Pair<T3, T4> & p);
};

template<class T1, class T2>
bool Pair<T1, T2>::operator < (const Pair<T1, T2> & p) const {
	return key < p.key;
}

template<class T1,class T1>
ostream & operator<<(ostream & o, const Pair<T1, T2> & p){
	o << "(" << p.key << ", " << p.value << ")";
	return o;
}

int main() {
	Pair<string, int> student("Tom", 19);
	Pair<int, double> obj(12, 3.14);
	cout << student.key << " " << student.value << endl; //("Tom", 19) (12, 3.14)
	
	return 0;
}

```

#### 7.2.3 函数模板作为类的友元

```c++
class A{
    int v;
  public:
    A(int n):v(n) { }
    template<class T>
    friend void Print(const T & p);
};

template<class T>
void Print(const T & p){
    cout << p.v;
}

int main(){
    A a(4);
    Print(a); //4
    return 0;
}
```

#### 7.2.4 类模板作为类模板的友元

```c++
template<class T>
class B{
    T v;
  public:
    B(T n):v(n) { }
    template<class T2>
    friend class A;
};

template<class T>
class A{
  public:
    void Func(){
        B<int> o(10);
        cout << o.v << endl; 
    }
};

int main(){
    //A<double>类，成了B<int>类的友元。任何从A模板实例化出来的类，都是任何B实例化出来的类的友元
    
    A<double> a;
    a.Func(); //10
    
    return 0;
}
```

### 7.3 类模板与静态成员变量

> 类模板中可以定义静态成员，那么从该类模板实例化得到的所有类，都包含同样的静态成员。

```c++
template<class T>
class A {
private:
	static int count;
public:
	A() { count++; }
	~A() { count--; }
	A(A&) { count++ }
	static void PrintCount() { cout << count << endl; }
};

template<> int A<int>::count = 0;
template<> int A<double>::count = 0;

int main() {
	A<int> ia;
	A<double> da;
	ia.PrintCount(); //1
    
	da.PrintCount(); //1
    
	return 0;
}
```

## 八、测验

### 8.1  简单的SumArray


  ```c++
#include <iostream>
#include <string>
using namespace std;
template <class T>
T SumArray(
//your code starts here

T * begin, T * end) {
	T tmp = * begin;
	++ begin;
	for(;  begin != end ; ++ begin)
		tmp += * begin;
	return tmp;
//your code ends here	

}
int main() {
	string array[4] = { "Tom","Jack","Mary","John"};
	cout << SumArray(array,array+4) << endl;
	int a[4] = { 1, 2, 3, 4};  //提示：1+2+3+4 = 10
	cout << SumArray(a,a+4) << endl;
	return 0;
} 
  ```

### 8.2  你真的搞清楚为啥 while(cin >> n) 能成立了吗？

```c++
#include <iostream>
using namespace std;
class MyCin
{
	//your code starts here
    
    bool valid;    
    public:
        MyCin():valid(true) { }
        operator bool( ) { //重载类型强制转换运算符 bool
            
            return valid; 
        }
        MyCin & operator >> (int & n)
        {
            cin >> n;
            if( n == -1 )
            	valid = false;
            
            return * this;
        }
//your code ends here     
    
};
int main()
{
    MyCin m;
    int n1,n2;
    while( m >> n1 >> n2) 
        cout  << n1 << " " << n2 << endl;
    return 0;
}
```

### 8.3 这个模板并不难

```c++
#include <iostream>
#include <string>
#include <cstring>
using namespace std;

template <class T>  
class myclass { 
//your code starts here
    
	T * p;;
	int size;
public:
	myclass ( T a [], int n)  
	{
		p = new T[n];
		for( int i = 0;i < n;i ++ )
			p[i] = a[i];
		size = n;
	} 
//your code ends here	
    
	~myclass( ) {
		delete [] p;
	}
	void Show()
	{
		for( int i = 0;i < size;i ++ ) {
			cout << p[i] << ",";
		}
		cout << endl;
	}
};
int a[100];
int main() {
	char line[100];
	while( cin >> line ) {
		myclass<char> obj(line,strlen(line));;
		obj.Show();
		int n;
		cin >> n;
		for(int i = 0;i < n; ++i)
			cin >> a[i];
		myclass<int> obj2(a,n);
		obj2.Show();
	}
	return 0;
}
```

### 8.4 排序，又见排序!

```c++
#include <iostream>
using namespace std;

bool Greater2(int n1,int n2) 
{
	return n1 > n2;
}
bool Greater1(int n1,int n2) 
{
	return n1 < n2;
}
bool Greater3(double d1,double d2)
{
	return d1 < d2;
}

template <class T1,class T2>
void mysort(
//your code starts here
    
 T1 * start , T1 * end, T2 myless )
{
	int size = end - start;
	for( int i = size -1;i >= 0 ; --i ) {
		for( int j = 0; j < i ; ++j ) {
			if(  myless( start[j+1],start[j] )) {
				T1 tmp = start[j];
				start[j] = start[j+1];
				start[j+1] = tmp; 
			}
		}
	}
}
/*答案2 ：
template<class T>
void Swap( T & a, T & b)
{
	T tmp;
	tmp = a;
	a = b;
	b = tmp;
}

template <class T1,class T2>
void mysort( T1  start , T1  end, T2 myless )
{
	int size = end - start;
	for( int i = size -1;i >= 0 ; --i ) {
		for( int j = 0; j < i ; ++j ) {
			if(  myless( * ( start + j+1), * (start+j) )) {
				Swap(* ( start + j+1), * (start+j) );
			}
		}
	}
}
答案 3 
template <class T1,class T2>
void mysort( T1  start , T1  end, T2 myless )
{
	int size = end - start;
	for( int i = size -1;i >= 0 ; --i ) {
		for( int j = 0; j < i ; ++j ) {
			if(  myless( * ( start + j+1), * (start+j) )) {
				auto tmp = * ( start+j);
				* ( start +j ) = * ( start + j+1);
				* ( start + j+1)  = tmp; 
			}
		}
	}
}

*/

//your code ends here

#define NUM 5
int main()
{
    int an[NUM] = { 8,123,11,10,4 };
    mysort(an,an+NUM,Greater1); //从小到大排序 
    for( int i = 0;i < NUM; i ++ )
       cout << an[i] << ",";
    mysort(an,an+NUM,Greater2); //从大到小排序 
    cout << endl;
    for( int i = 0;i < NUM; i ++ )
        cout << an[i] << ","; 
    cout << endl;
    double d[6] = { 1.4,1.8,3.2,1.2,3.1,2.1};
    mysort(d+1,d+5,Greater3); //将数组从下标1到下标4从小到大排序 
    for( int i = 0;i < 6; i ++ )
         cout << d[i] << ","; 
	return 0;
}

```

### 8.5 山寨版istream_iterator

```c++
#include <iostream>
#include <string>

using namespace std;
template <class T>
class CMyistream_iterator
{
//your code starts here	
    
	istream &  r;
	T v;
	public:
		T operator *() {
			return v;	
		}
		CMyistream_iterator( istream & rr):r(rr) {
			r >> v ;
		}
		void operator ++ (int) {
			 r >> v ;
		}
//your code ends here	
    
};



int main()  
{ 
	int t;
	cin >> t;
	while( t -- ) {
		 CMyistream_iterator<int> inputInt(cin);
		 int n1,n2,n3;
		 n1 = * inputInt; //读入 n1
        
		 int tmp = * inputInt;
		 cout << tmp << endl;
		 inputInt ++;   
		 n2 = * inputInt; //读入 n2
        
		 inputInt ++;
		 n3 = * inputInt; //读入 n3
        
		 cout << n1 << " " << n2<< " " << n3 << " ";
		 CMyistream_iterator<string> inputStr(cin);
		 string s1,s2;
		 s1 = * inputStr;
		 inputStr ++;
		 s2 = * inputStr;
		 cout << s1 << " " << s2 << endl;
	}
	 return 0;  
}
```

