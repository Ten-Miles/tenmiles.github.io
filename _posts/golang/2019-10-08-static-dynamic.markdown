---
layout:     post
title:      "Golang中的静态类型与动态类型"
subtitle:   ""
date:       2019-10-08 12:00:00
author:     "zhihao"
header-img: "img/post-bg-css.jpg"
tags:
    - golang
---

## 静态类型
>所谓的静态类型（即 static type），就是变量声明的时候的类型

```golang
var age int   // int 是静态类型
var name string  // string 也是静态类型
```

## 动态类型
>所谓的 动态类型（即 concrete type，也叫具体类型）是 程序运行时系统才能看见的类型

```golang
var i interface{}
i = 99
i = "动态类型"
```
第一行：我们在给 i 声明了 interface{} 类型，所以 i 的静态类型就是 interface{}

第二行：当我们给变量 i 赋一个 int 类型的值时，它的静态类型还是 interface{}，这是不会变的，但是它的动态类型此时变成了 int 类型。

第三行：当我们给变量 i 赋一个 string 类型的值时，它的静态类型还是 interface{}，它还是不会变，但是它的动态类型此时又变成了 string 类型。

根据接口是否包含方法，可以将接口分为 `iface` 和 `eface`

`iface`表示带有一组方法的接口
```golang
type Phone interface {
   call()
}
```
源码如下：
```golang
// runtime/runtime2.go
// 非空接口
type iface struct {
    tab  *itab
    data unsafe.Pointer
}

// 非空接口的类型信息
type itab struct {
    inter  *interfacetype  // 接口定义的类型信息
    _type  *_type      // 接口实际指向值的类型信息
    link   *itab
    bad    int32
    inhash int32
    fun    [1]uintptr   // 接口方法实现列表，即函数地址列表，按字典序排序
}

// runtime/type.go
// 非空接口类型，接口定义，包路径等。
type interfacetype struct {
   typ     _type
   pkgpath name
   mhdr    []imethod      // 接口方法声明列表，按字典序排序
}
// 接口的方法声明
type imethod struct {
   name nameOff          // 方法名
   ityp typeOff                // 描述方法参数返回值等细节
}
```
`eface`表示不带有方法的接口
```golang
var i interface{}
```
源码如下：
```golang
// src/runtime/runtime2.go
// 空接口
type eface struct {
    _type *_type
    data  unsafe.Pointer
}
```
## 三个细节
#### 1.对方法的调用限制
接口是一组固定的方法集，由于静态类型的限制，接口变量有时仅能调用其中特定的一些方法。

```golang
package main

import "fmt"

type Phone interface {
    call()
}

type iPhone struct {
    name string
}

func (phone iPhone)call()  {
    fmt.Println("Hello, iPhone.")
}

func (phone iPhone)send_wechat()  {
    fmt.Println("Hello, Wechat.")
}

func main() {
    var phone Phone
    phone = iPhone{name:"ming's iphone"}
    phone.call()
    phone.send_wechat()
}
```
这里调用 `phone.send_wechat`方法，程序会报错，提示我们 Phone 类型的方法没有 send_wechat 的字段或方法。

原因也很明显，因为我们的phone对象显示声明为 Phone 接口类型，因此 phone调用的方法会受到此接口的限制。

**那么如何让phone 可以调用 send_wechat 方法呢？** 

答案是可以不显示的声明为 Phone接口类型 ，但要清楚 phone 对象实际上是隐式的实现了 Phone 接口，如此一来，方法的调用就不会受到接口类型的约束。

修改 main 方法成如下
```golang
func main() {
    phone := iPhone{name:"ming's iphone"}
    phone.call()
    phone.send_wechat()
}
```
此时可以正常运行。

#### 2. 调用函数时的隐式转换

Go 语言中的函数调用都是值传递的，变量会在方法调用前进行类型转换。

比如下面这段代码
```golang
import (
    "fmt"
)

func printType(i interface{})  {

    switch i.(type) {
    case int:
        fmt.Println("参数的类型是 int")
    case string:
        fmt.Println("参数的类型是 string")
    }
}

func main() {
    a := 10
    printType(a)
}
```
此时正常运行，输出为
```
参数的类型是 int
```

但是如果你把函数内的内容搬到到外面来
```golang
package main

import "fmt"


func main() {
    a := 10

    switch a.(type) {
    case int:
        fmt.Println("参数的类型是 int")
    case string:
        fmt.Println("参数的类型是 string")
    }
}
```
此时会报错
```
# command-line-arguments
./demo.go:9:5: cannot type switch on non-interface value a (type int)
```
这是因为当一个函数接口 interface{} 空接口类型时，我们说它可以接收什么任意类型的参数，当你使用这种写法时，Go 会默默地为我们做一件事，就是把传入函数的参数值的类型隐式的转换成 interface{} 类型。

如果你想手动对其进行类型转换，可以像下面这样子，就可以将变量 a 的静态类型转换为 interface{} 类型然后赋值给 b （此时 a 的静态类型还是 int，而 b 的静态类型为 interface{}）
```golang
var a int = 25
b := interface{}(a)
```
原来的代码可以像上述这样修改：
```golang
package main

import "fmt"

func main() {
    a := 10

    switch interface{}(a).(type) {
    case int:
        fmt.Println("参数的类型是 int")
    case string:
        fmt.Println("参数的类型是 string")
    }
}
```
#### 3. 类型断言中的隐式转换
上面我们知道了，只有静态类型为接口类型的对象才可以进行类型断言。

而当类型断言完成后，会返回一个静态类型为你断言的类型的对象，也就是说，当我们使用了类型断言，Go 实际上又会默认为我们进行了一次隐式的类型转换。

验证方法也很简单，使用完一次类型断言后，对返回的对象再一次使用类型断言，Goland 立马就会提示我们新对象 b 不是一个接口类型的对象，不允许进行类型断言。