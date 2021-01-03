---
layout:     post
title:      Unity Shader
subtitle:   Unity Shader结构
date:       2021-01-03
author:     LudoArt
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Unity Shader
    - 游戏开发
---

# Unity Shader结构

基础结构

```
Shader "ShaderName"{
	Properties{
		// 属性
	}
	SubShader{
		// 显卡A使用的子着色器
	}
	SubShader{
		// 显卡B使用的子着色器
	}
	Fallback "VertexLit"
}
```

## Shader的名字

每个Unity Shader文件的第一行都需要通过Shader语义来指定该Unity Shader的名字。当为材质选择使用的Unity Shader时，这些名称就会出现在材质面板的下拉列表里。通过在字符串中添加斜杠，可以控制Unity Shader在材质面板中出现的位置。如：

`Shder "Custom/MyShader" { }`

那么这个Unity Shader在材质面板中的位置就是：Shader -> Custom -> MyShader。

## Properties属性

Properties语义块中包含了一系列属性，这些属性将会出现在材质面板中。

Properties语义块的**定义**通常如下：

```
Properties{
	Name ("display name", PropertyType) = DefaultValue
	Name ("display name", PropertyType) = DefaultValue
	// 更多属性
}
```

- `Name`：属性的名字，通过由一个下划线开始；
- `display name`：出现在材质面板上的名字；
- `PropertyType`：属性类型，常见的属性类型见下表；
- `DefaultValue`：属性默认值。

| 属性类型        | 默认值的定义语法 | 例子 |
| :-------------- | :--------------- | :--- |
| Int             | number           | _Int ("Int", Int) = 2 |
| Float           | number           | _Float ("Float", Float) = 1.5 |
| Range(min, max) | number           | _Range ("Range", Range(0.0, 5.0)) = 3.0 |
| Color           | (number,number,number,number) | _Color (“Color”, Color) = (1,1,1,1) |
| Vector          | (number,number,number,number) | _Vector ("Vector", Vector) = (2,3,6,1) |
| 2D              | "defaulttexture" {} | _2D ("2D", 2D) = "" {} |
| Cube            | "defaulttexture" {} | _Cube ("Cube", Cube) = "white" {} |
| 3D              | "defaulttexture" {} | _3D ("3D", 3D) = "black" {} |

## SubShader

每一个Unity Shader文件可以包含多个SubShader语义块，但最少要有一个。当Unity需要加载这个Unity Shader时，Unity会扫描所有的SubShader语义块，然后选择第一个能够在目标平台上运行的SubShader。如果都不支持的话，Unity就会使用Fallback语义指定的Unity Shader。

SubShader语义块中包含的**定义**通常如下：

```
SubShader{
	// 可选的
	[Tags]
	
	// 可选的
	[RenderSetup]
	
	Pass{
	}
	// Other Passes
}
```

**状态[RenderSetup]**和**标签[Tags]**可以在SubShader中声明，也可以在Pass声明，不同的是，SubShader中的一些标签设置是特定的，即这些标签设置和Pass中使用的标签是不一样的。

### 状态设置

下表给出了ShaderLab中常见的渲染状态设置选项。

| 状态名称 | 设置指令                                                     | 解释                                 |
| -------- | ------------------------------------------------------------ | ------------------------------------ |
| Cull     | Cull Back \| Front \| Off                                    | 设置剔除模式：剔除背面/正面/关闭剔除 |
| ZTest    | ZTest Less Greater \| LEqual \| GEqual \| Equal \| NotEqual \| Always | 设置深度测试时使用的函数             |
| ZWrite   | ZWrite On \| Off                                             | 开启/关闭深度写入                    |
| Blend    | Blend SrcFactor DstFactor                                    | 开启并设置混合模式                   |

### SubShader标签

SubShader的标签是一个键值对，它的键和值都是字符串类型。它们用来告诉渲染引擎：我们希望怎样以及何时渲染这个对象。

标签的结构如下：

`Tags { "TagName1" = "Value1" "TagName2" = "Value2" }`

SubShader的标签块支持的标签类型如下表所示：

| 标签类型             | 说明                                                         | 例子                                     |
| -------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| Queue                | 控制渲染顺序，指定该物体属于哪一个渲染队列                   | Tags { "Queue" = "Transparent" }         |
| RenderType           | 对着色器进行分类                                             | Tags { "RenderType" = "Opaque" }         |
| DisableBatching      | 直接指明是否对该SubShader使用批处理                          | Tags { "DisableBatching" = "True" }      |
| ForceNoShadowCasting | 控制使用该SubShader的物体是否会投射阴影                      | Tags { "ForceNoShadowCasting" = "True" } |
| IgnoreProjector      | 如果该标签值为True，那么使用该SubShader的物体将不会受Projector的影响（通常用于半透明物体） | Tags { "IgnoreProjector" = "True" }      |
| CanUseSpriteAtlas    | 当该SubShader是用于精灵时，将该标签设为False                 | Tags { "CanUseSpriteAtlas" = "False" }   |
| PreviewType          | 指明材质面板将如何预览该材质（默认将显示为一个球形）         | Tags { "PreviewType" = "Plane" }         |

上述标签仅可以在SubShader中声明，而不可以在Pass块中声明。

 ### Pass语义块

Pass语义块包含的语义如下：

```
Pass{
	[Name]
	[Tags]
	[RenderSetup]
	// Other code
}
```

- `Name`：名称
- `RenderSetup`：渲染状态。SubShader的渲染状态同样适用于Pass，此外，在Pass中还可以使用固定管线的着色器命令。
- `Tags`：标签。不同于SubShader的标签。这些标签也是用于告诉渲染引擎我们希望怎样来渲染该物体。

| 标签类型       | 说明 | 例子 |
| -------------- | ---- | ---- |
| LightMode      | 定义该Pass在Unity的渲染流水线中的角色 | Tags { "LightMode" = "ForwardBase" } |
| RequireOptions | 用于指定当满足某些条件时才渲染该Pass | Tags { "RequireOptions" = "SoftVegetation" } |

除了上面普通的Pass定义外，Unity Shader还支持一些特殊的Pass，以便进行代码复用或实现更复杂的效果。

- **UsePass**：可以使用该命令来复用其他Unity Shader中的Pass；
- **GrabPass**：该Pass负责抓取屏幕并将结果存储在一张纹理中，以用于后续的Pass处理。

## Fallback

紧跟在各个SubShader语义块后面的，可以是一个Fallback指令，用于以上SubShader均不可运行的情况。

Fallback的语义如下：

```
Fallback "name"
// 或者
Fallback Off
```

# Unity Shader形式

在Unity中，我们可以使用下面3种形式来编写Unity Shader。

而不管使用哪种形式，真正意义上的Shader代码都需要包含在ShaderLab语义块中，如下所示：

```
Shader "MyShader"{
	Properties{
		// 所需的各种属性
	}
	SubShader{
		// 真正意义上的Shader代码会出现在这里
		// 表面着色器（Surface Shader）或者
		// 顶点/片元着色器（Vertex/Fragment Shader）或者
		// 固定函数着色器（Fixed Function Shader）
	}
	SubShader{
		// 和上一个SubShader类似
	}
}
```

## 表面着色器

表面着色器被定义在SubShader语义块中的CGPROGRAM和ENDCG之间。

## 顶点/片元着色器

顶点/片元着色器被定义在Pass语义块中的CGPROGRAM和ENDCG之间。

## 固定函数着色器

固定函数着色器被定义在Pass语义块中，这些着色器往往只能完成一些非常简单的效果。

> **扩展阅读：**
>
> Unity官方文档：[Unity官方文档](https://docs.unity3d.com/Manual/SL-Reference.html)
>
> Unity简单的着色器编写教程：[教程1](https://docs.unity3d.com/Manual/ShaderTut1.html)，[教程2](https://docs.unity3d.com/Manual/ShaderTut2.html)
>
> NVIDIA的Cg系列教程：[Cg系列教程](http://http.developer.nvidia.com/CgTutorial/cg_tutorial_chapter01.html)