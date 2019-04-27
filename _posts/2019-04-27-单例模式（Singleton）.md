---
layout:     post
title:      单例模式（Singleton）
subtitle:   null
date:       2019-04-27
author:     LudoArt
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 设计模式
---

# 单例模式——对象创建型模式

## 1、意图

保证一个类仅有一个实例，并提供一个访问它的全局访问点。

## 2、适用性

在以下任一情况下可以使用单例模式：

- 当类只能有一个实例而且客户可以从一个众所周知的访问点访问它时。
- 当这个唯一实例应该是通过子类化可扩展的，并且客户应该无需更改代码就能使用一个扩展的实例时。

### 3、结构

![观察者模式结构](https://i.postimg.cc/ZRjF5rp0/image.png)

<center>单例模式结构</center>

## 4、参与者

- **Singleton**
  - 定义一个Instance操作，允许客户访问它的唯一实例。Instance是一个类操作。
  - 可能负责创建它自己的唯一实例。

## 5、协作

- 客户只能通过Singleton的Instance操作访问一个Singleton的实例。

## 6、效果

单例模式的一些优点：

1. **对唯一实例的受控访问**

   因为Singleton类封装它的唯一实例，所以它可以严格的控制客户怎样以及何时访问它。

2. **缩小名空间**

   Singleton模式是对全局变量的一种改进。它避免了那些存储唯一实例的全局变量污染名空间。

3. **允许对操作和表示的精化**

   Singleton类可以有子类，而且用这个扩展类的实例来配置一个应用是很容易的。你可以用你所需要的类的实例在运行时刻配置应用。

4. **允许可变数目的实例**

   允许Singleton类有多个实例。此外，可以用相同的方法来控制应用所使用的实例的数目。只有允许访问Singleton实例的操作需要改变。

## 7、实现

1. **保证一个唯一的实例**

   常用方法是将创建这个实例的操作隐藏在一个类操作（即一个静态成员函数或者是一个类方法）后面，由它保证只有一个实例被创建，这个操作可以访问保存唯一实例的变量，而且它可以保证这个变量在返回值之前用这个唯一实例初始化。

   在C++中，可以用Singleton类的静态成员函数Instance来定义这个类操作，Singleton还定义了一个静态成员变量_instance，它包含了一个指向它的唯一实例的指针。                                                                                                                                                                                                                                                                 

2. **创建Singleton类的子类**

   目标对象可以简单地将自己作为Update操作的一个参数，让观察者知道应去检查哪一个目标。

## 8、代码示例

```c++
class MazeFactory {
public:
	static MazeFactory* Instance();

protected:
	MazeFactory();

private:
	static MazeFactory* _instance;
};

MazeFactory* MazeFactory::_instance = 0;

//假定不生成MazeFactory的子类

MazeFactory* MazeFactory::Instance() {
	if (_instance == 0) {
		_instance = new MazeFactory;
	}
	return _instance;
}

// 考虑当存在MazeFactory的多个子类

MazeFactory* MazeFactory::Instance() {
	if (_instance == 0) {
		const char* mazeStyle = getenv("MAZESTYLE");// 该函数具体作用见下文

		if (strcmp(mazeStyle, "bombed") == 0) {
			_instance = new BombedMazeFactory;
		}else if(strcmp(mazeStyle, "enchanted") == 0) {
			_instance = new EnchantedMazeFactory;
		}
		// ... other possible subclasses
        
		else {
			//default
            
			_instance = new MazeFactory;
		}
	}
	return _instance;
}
```



**PS：`getenv` 函数的作用**

|          | getenv(取得环境变量内容)                                     |
| :------: | :----------------------------------------------------------- |
| 相关函数 | putenv，setenv，unsetenv                                     |
| 表头文件 | #include<stdlib.h>                                           |
| 定义函数 | char * getenv(const char *name);                             |
| 函数说明 | getenv()用来取得参数name环境变量的内容。参数name为环境变量的名称，如果该变量存在则会返回指向该内容的指针。环境变量的格式为name＝value。 |
|  返回值  | 执行成功则返回指向该内容的指针，找不到符合的环境变量名称则返回NULL。 |
|   范例   | #include<stdlib.h><br/>mian()<br/>{<br/>char *p;<br/>if((p = getenv(“USER”)))<br/>printf(“USER=%s/n”,p);<br/>} |
| 运行结果 | USER = root                                                  |
