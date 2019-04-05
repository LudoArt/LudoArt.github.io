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



## 三、文件读写（一）



## 四、文件读写（二）



## 五、函数模版



## 六、类模版



## 七、类模版与派生、友元和静态成员变量



## 八、测验

