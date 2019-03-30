---
layout:     post
title:      C++ Study Week Four
subtitle:   null
date:       2019-03-30
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

# 第四周

## 一、运算符重载的基本概念

> - 运算符重载的实质是函数重载
> - 可以重载为普通函数，也可以重载为成员函数
> - 把含运算符的表达式转换成对运算符函数的调用
> - 把运算符的操作数转换成运算符函数的参数
> - 运算符被多次重载时，根据实参的类型决定调用哪个运算符函数

运算符重载的形式：

返回值类型 operator 运算符（形参表）

{

​	……

}

```c++
class Complex{
  public:
    double real,imag;
    Complex(double r = 0.0, double i = 0.0):real(r),imag(i){}
    Complex operator- (const Complex & c);
};
//重载为普通函数时，参数个数为运算符目数
Complex operator+ (const Complex & a, const Complex & b){
    return Complex(a.real + b.real, a.imag + b.imag); //返回一个临时对象
}
//重载为成员函数时，参数个数为运算符目数减一
Complex operator- (const Complex & c)){
    return Complex(real - c.real, imag - c.imag); //返回一个临时对象
}
```

## 二、赋值运算符的重载

> 赋值运算符 “=” 只能重载为成员函数

```c++
class String{
    private:
    char * str;
    public:
    String ():str(new char[1]) { str[0] = 0; }
    const char * c_str() { return str; }
    String & operator = ( const char * s );
    String::~String() { delete [] str; }
};
String & String::operator = (const char * s){
    delete [] str;
    str = new char[strlen(s) + 1];
    strcpy(str, s);
    return * this;
}
int main(){
    String s;
    s = "Goog luck."; //等价于s.operator = ("Goog luck.");
    cout << s.c_str() << endl; //Goog luck.
    String s2 = "hello!"; //error，该句不是赋值语句，该句应调用构造函数
    s = "hello!"; //等价于s.operator = ("hello!");
    cout << s.c_str() << endl; //hello!
    return 0;
}
```

```c++
//赋值运算符的改进1
String s1, s2;
s1 = "this";
s2 = "that";
s1 = s2; //出错，此时s1和s2指向同一片内存地址，危险（浅拷贝）

//改进方法
String & String::operator = (const String & s){
    delete [] str;
    str = new char[strlen(s) + 1];
    strcpy(str, s);
    return * this;
}

//赋值运算符的改进2
String s;
s = "Hello";
s = s; //出错

//改进方法（深拷贝）
String & String::operator = (const String & s){
    if (this == & s)
        return * this;
    delete [] str;
    str = new char[strlen(s) + 1];
    strcpy(str, s);
    return * this;
}
```

## 三、运算符重载为友元

```c++
class Complex{
  public:
    double real,imag;
    Complex(double r = 0.0, double i = 0.0):real(r),imag(i){}
    Complex operator+ (double r){
        //能解释 c + 5
        return Complex(real + r, imag);
    }
};
int main(){
    Complex c;
    c = c + 5; //相当于c = c.operator+ (5);
    c = 5 + c; //error
}

//解决方法：在Complex类中添加一个友元函数
 friend Complex operator+ (double r, const Complex & c){
        //能解释 5 + c
        return Complex(c.real + r, c.imag);
    }
```

## 四、可变长数组类的实现

简易版vector类的实现（可自己动手尝试一下）

## 五、流插入运算符和流提取运算符的重载

```c++
class Complex{
    double real,imag;
  public:
    Complex(double r = 0.0, double i = 0.0):real(r),imag(i){};
    friend ostream & operator<<(ostream & os, const Complex & c);
    friend istream & operator>>(istream & is, Complex & c);
};

ostream & operator<<(ostream & os, const Complex & c){
    os << c.real << "+" << c.imag << "i"; //以“a+bi”的形式输出
    return os;
}

istream & operator>>(istream & is, Complex & c){
    string s;
    is >> s; //将“a+bi”作为字符串读入，“a+bi”中间不能有空格
    int pos = s.find("+", 0);
    string sTmp = s.substr(0, pos); //分离出代表实部的字符串
    c.real = atof(sTmp.c_str()); //atof库函数能够将const char*指针指向的内容转换成float
    sTmp = s.substr(pos + 1, s.length() - pos - 2); //分离出代表虚部的字符串
    c.imag = atof(sTmp.c_str());
    return is;
}
```

## 六、类型转换运算符的重载

```c++
class Complex{
    double real,imag;
  public:
    Complex(double r = 0.0, double i = 0.0):real(r),imag(i){};
    operator double() { return real; }
    //重载强制类型转换运算符double
};

int main(){
    Complex c(1.2, 3.4);
    cout << (double)c << endl; //输出1.2
    double n = 2 + c; //等价于 double n = 2 + c.operator double()
    cout << n; //输出3.2
}
```

## 七、自增自减运算符的重载

> - **前置运算符作为一元运算符重载**
>   - 重载为成员函数：
>   	- `T & operator++();`
>   	- `T & operator--();`
>   - 重载为全局函数：
>    	- `T1 & operator++(T2);`
>   	- `T1 & operator--(T2);`
> - **后置运算符作为二元运算符重载（多写一个没用的参数）**
>   - 重载为成员函数：
>     - `T operator++(int);`
>     - `T operator--(int);`
>   - 重载为全局函数：
>     - `T1 operator++(T2, int);`
>     - `T1 operator--(T2, int);`

```c++
class CDmeo{
  private:
    int n;
  public:
    CDmeo(int i = 0):n(i) { }
    CDmeo & operator++ (); //用于前置形式
    CDmeo operator++ (int); //用于后置形式
    operator int () { return n; }
    friend CDmeo & operator-- (CDmeo & );
    friend CDmeo operator-- (CDmeo &, int);
};
CDmeo & CDmeo::operator++ (){
    //前置++
    ++n;
    return * this;
    //++s即为：s.:operator++();
}
CDmeo CDmeo::operator++ (int k){
    //后置++
    CDmeo tmp(*this); //记录修改前的对象
    n++;
    return tmp; //返回修改前的对象
    //s++即为：s.:operator++(0);
}
CDmeo & CDmeo::operator-- (CDmeo & d){
    //前置--
    d.n--;
    return d;
    //--s即为：operator--(s);
}
CDmeo & CDmeo::operator-- (CDmeo & d, int){
    //后置--
    CDemo tmp(d);
    d.n--;
    return tmp;
    //s--即为：operator--(s, 0);
}
```

## 八、测验

### 一、第四周程序填空题3

```c++
/*
用一位数组来存放二维数组
a[i][j]的计算过程从左到右，a[i]的返回值是个指针，指向第i行的首地址
a[i][j]就会是第i行第j列的元素了
*/
class Array2 {
	// 在此处补充你的代码
private:
	int *p;
	int r, c;
public:
	Array2() { p = NULL; }
	Array2(int m, int n):r(m),c(n) {
		p = new int[r*c];
	}
	Array2(Array2 & a) :r(a.r), c(a.c) {
		p = new int[r*c];
		memcpy(p, a.p, sizeof(int)*r*c);
	}
	Array2 & operator=(const Array2 & a) {
		if (p)
			delete[]p;
		r = a.r;
		c = a.c;
		p = new int[r*c];
		memcpy(p, a.p, sizeof(int)*r*c);
		return *this;
	}
	~Array2() {
		if (p)
			delete[]p;
	}
	int * operator[](int i) {
		return p + i * c;
	}
	int & operator()(int i,int j) {
		return p[i*c + j];
	}
};

int main() {
	Array2 a(3, 4);
	int i, j;
	for (i = 0; i < 3; ++i)
		for (j = 0; j < 4; j++)
			a[i][j] = i * 4 + j;
	for (i = 0; i < 3; ++i) {
		for (j = 0; j < 4; j++) {
			cout << a[i][j] << ",";
		}
		cout << endl;
	}
	cout << "next" << endl;
	Array2 b;     
	b = a;
	for (i = 0; i < 3; ++i) {
		for (j = 0; j < 4; j++) {
			cout << b[i][j] << ",";
		}
		cout << endl;
	}
	return 0;
}
```

### 二、别叫，这个大整数已经很简化了!

```c++
const int MAX = 110; 
class CHugeInt {
private:
    char maxNum[210];
    int len;
public:
    CHugeInt(char * s){
        strcpy(maxNum,s);
        int i=0,j=strlen(s)-1;
        while(i<j)
        {
            swap(maxNum[i],maxNum[j]);
            i++;
            j--;
        }
        len=strlen(s);
    }
    CHugeInt(){
        len=0;
    } 
    CHugeInt(int n){
        int i=0;
        if(n==0)
        {
            maxNum[i++]='0';
        }else{
            while(n)
            {
                maxNum[i++]=n%10+'0';
                n=n/10;
            }    
        }
        maxNum[i]='\0';
        len=i;
    }
    CHugeInt  operator+(CHugeInt & a)
    {
            int i=0,j=0;
            int t,sum=0;
            CHugeInt temps;
            strcpy(temps.maxNum,maxNum);
            temps.len=len;
            int flag=0;
            while(j<a.len&&i<temps.len)
            {
                t=a.maxNum[j]-'0';
                int te=temps.maxNum[i]-'0';
                sum=t+te;
                if(sum>=10)
                {
                    temps.maxNum[i]=sum%10+'0';
                    temps.maxNum[i+1]=sum/10+temps.maxNum[i+1];
                    if(i+1>=temps.len)
                    {
                        temps.maxNum[i+1]+='0'; 
                    }
                    flag=1;
                }else{
                    flag=0;
                    temps.maxNum[i]=sum+'0';
                }
                i++,j++;
                sum=0;
            }
            while(j<a.len)
            {
                if(flag==1)
                {
                    temps.maxNum[i+1]=a.maxNum[j];
                    i++,j++;    
                }else{
                    temps.maxNum[i]=a.maxNum[j];
                    i++,j++;
                }
            }
            if(i>=len)
            {
                if(flag==1){
                    temps.maxNum[i+1]='\0';
                    temps.len=i+1;
                }
                else{
                    temps.maxNum[i]='\0';
                    temps.len=i;
                }        
            }
        return temps;
    }
    CHugeInt & operator +=(int n)
    {
        CHugeInt temps(n);
        *this=this->operator+(temps);
        return *this;
    }
    friend ostream & operator<<(ostream & os,const CHugeInt & s)
    {
            int i=0,j=s.len-1;
            for(;j>=i;j--)
                os<<s.maxNum[j];
            return os;
    }
    friend CHugeInt  operator+(int n,CHugeInt  s)
    {
        CHugeInt temps(n);
        s=s+temps;
        return s;
    }
    friend CHugeInt  operator+(CHugeInt  s,int n)
    {
        CHugeInt temps(n);
        s=s+temps;
        return s;
    }
    CHugeInt &  operator++()
    {
        (*this)+=1;
        return *(this);
    }
    CHugeInt   operator++(int n)
    {
        CHugeInt temps;
        strcpy(temps.maxNum,maxNum);
        temps.len=len;
        this->operator +=(1);
        return temps;
    }
};
int  main() 
{ 
	char s[210];
	int n;

	while (cin >> s >> n) {
		CHugeInt a(s);
		CHugeInt b(n);

		cout << a + b << endl;
		cout << n + a << endl;
		cout << a + n << endl;
		b += n;
		cout  << ++ b << endl;
		cout << b++ << endl;
		cout << b << endl;
	}
	return 0;
}
```

