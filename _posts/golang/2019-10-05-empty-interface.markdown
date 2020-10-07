---
layout:     post
title:      "Golang中的空接口"
subtitle:   ""
date:       2019-10-05 12:00:00
author:     "zhihao"
header-img: "img/post-bg-css.jpg"
tags:
    - golang
---


> 空接口是特殊形式的接口类型，普通的接口都有方法，而空接口没有定义任何方法口，也因此，我们可以说所有类型都至少实现了空接口。
## 如何使用空接口
第一，通常我们会直接使用 `interface{}` 作为类型声明一个实例，而这个实例可以承载任意类型的值。
```golang
package main

import (
    "fmt"
)

func main()  {
    // 声明一个空接口实例
    var i interface{}

    // 存 int 没有问题
    i = 1
    fmt.Println(i)

    // 存字符串也没有问题
    i = "hello"
    fmt.Println(i)

    // 存布尔值也没有问题
    i = false
    fmt.Println(i)
}
```
第二，如果想让你的函数可以接收任意类型的值 ，也可以使用空接口

接收一个任意类型的值

```golang
package main

import (
    "fmt"
)

func myfunc(iface interface{}){
    fmt.Println(iface)
}

func main()  {
    a := 10
    b := "hello"
    c := true

    myfunc(a)
    myfunc(b)
    myfunc(c)
}
```

接收任意个任意类型的值

```golang
package main

import (
    "fmt"
)

func myfunc(ifaces ...interface{}){
    for _,iface := range ifaces{
        fmt.Println(iface)
    }
}

func main()  {
    a := 10
    b := "hello"
    c := true

    myfunc(a, b, c)
}
```
第三，你也定义一个可以接收任意类型的 array、slice、map、strcut，例如这边定义一个切片
```golang
package main

import "fmt"

func main() {
    any := make([]interface{}, 5)
    any[0] = 11
    any[1] = "hello world"
    any[2] = []int{11, 22, 33, 44}
    for _, value := range any {
        fmt.Println(value)
    }
}
```
## 注意事项

* 空接口可以承载任意值，但不代表任意类型就可以承接空接口类型的值

从实现的角度看，任何类型的值都满足空接口。因此空接口类型可以保存任何值，也可以从空接口中取出原值。

但要是你把一个空接口类型的对象，再赋值给一个固定类型（比如 int, string等类型）的对象赋值，是会报错的。

```golang
package main

func main() {
    // 声明a变量, 类型int, 初始值为1
    var a int = 1

    // 声明i变量, 类型为interface{}, 初始值为a, 此时i的值变为1
    var i interface{} = a

    // 声明b变量, 尝试赋值i
    var b int = i
}
```

* 当空接口承载数组和切片后，该对象无法再进行切片

```golang
package main

import "fmt"

func main() {
    sli := []int{2, 3, 5, 7, 11, 13}

    var i interface{}
    i = sli

    g := i[1:3]
    fmt.Println(g)
}
```
执行会报错。
```
.\main.go:11:8: cannot slice i (type interface {})
```

* 当你使用空接口来接收任意类型的参数时，它的静态类型是 `interface{}`，但动态类型（是 `int`，`string` 还是其他类型）我们并不知道，因此需要使用类型断言。

```golang
package main

import (
    "fmt"
)

func myfunc(i interface{})  {

    switch i.(type) {
    case int:
        fmt.Println("参数的类型是 int")
    case string:
        fmt.Println("参数的类型是 string")
    }
}

func main() {
    a := 10
    b := "hello"
    myfunc(a)
    myfunc(b)
}
```
输出结果：
```
参数的类型是 int
参数的类型是 string
```