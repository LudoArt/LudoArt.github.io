---
layout:     post
title:      new与malloc的区别
subtitle:   null
date:       2019-07-27
author:     LudoArt
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - C++
    - 面试经验

---


# new/delete与malloc/free的区别

首先，new/delete是C++的关键字，而malloc/free是C语言的库函数，后者使用必须指明申请内存空间的大小，对于类类型的对象，后者不会调用构造函数和析构函数。

其次，从源代码分析，对于

```c++
Complex* pc = new Complex(1, 2);
```

编译器将转化为（因编译器不同而有所不同，但基本思路一致）：

```c++
Complex *pc;

void* mem = operator new(sizeof(Complex)); //分配内存，其内部调用malloc(n)

pc = static_cast<Complex*>(mem);           //转型

pc->Complex::Complex(1, 2);                //构造函数

```
