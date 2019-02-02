---
layout:     post
title:      安装Git和TortoiseGit
subtitle:   null
date:       2019-01-12
author:     LudoArt
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 安装和配置环境
---
# 安装Git和TortoiseGit

## 安装Git

1. 下载最新版Git → [下载地址](<http://git-scm.com/download/win>)

2. 开始安装，一直点击Next（下一步）直到“Select Components”界面，如图所示，我们只需要勾选“Git Bash Here”

   ![select-Git-components-during-installation-guganeshan_thumb](http://guganeshan.com/blog/wp-content/uploads/2013/09/select-Git-components-during-installation-guganeshan.jpg)

3. 不用改变任何设置，一直点击“Next”即可，直到最后点击“Finish”完成安装。

## 配置Git设置
1. 打开“Git Bash"（可以在开始菜单中找到）

2. 配置用户名和邮箱地址（如配置用户名为John Doe，邮箱为johndoe@doebrothers.com，邮箱为github注册邮箱）

    ![setting-username-and-email-address-in-git](http://guganeshan.com/blog/wp-content/uploads/2013/09/setting-username-and-email-address-in-git-guganeshan.com_.jpg)

## 创建SSH身份
为了确保我们每次从Github存储库推送或拔出时都不输入用户名和密码，我们应该使用SSH（安全shell）与Github进行通信。
1. 打开“Git Bash"（如已打开请忽略）

2. 输入以下命令→ ` ssh-keygen –t rsa –C “johndoe@doebrothers.com”` 

   *这里的邮箱地址填你自己在Github上注册的邮箱*

3. 系统将提示您提供文件位置并输入两次密码。按住Enter键接受默认文件位置并跳过提供密码短语。

4. 现在生成密钥文件。将公钥内容复制到剪贴板的最简单方法是使用“clip”命令，如下→ `clip <〜/ .ssh / id_rsa.pub `

   按Enter键后，内容将显示在剪贴板中，可以粘贴。 如图

   ![create-ssh-identity](http://guganeshan.com/blog/wp-content/uploads/2013/09/create-ssh-identity-and-associate-with-bitbucket.jpg)

## 创建新的公共密钥

登录Github，新建一个公共密钥，将刚刚复制的密钥粘贴进去，如图

![](https://i.imgur.com/BBURFez.png)

![](https://i.imgur.com/ewXI1fV.png)

![](https://i.imgur.com/IJdgFDP.png)

## 安装TortoiseGit

1. 下载最新版的TortoiseGit → [下载地址](<http://code.google.com/p/tortoisegit/>)

2.  在选择SSH客户端（Choose SSH client）的时候，选择“OpenSSH, Git default SSH Client”，如图
![](http://guganeshan.com/blog/wp-content/uploads/2013/09/choose-ssh-client.jpg)

3. 点击“Finish”结束安装。
