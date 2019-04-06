---
layout:     post
title:      C++ Study Week Seven
subtitle:   null
date:       2019-04-06
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
```

## 三、文件读写（一）



## 四、文件读写（二）



## 五、函数模版



## 六、类模版



## 七、类模版与派生、友元和静态成员变量



## 八、测验

