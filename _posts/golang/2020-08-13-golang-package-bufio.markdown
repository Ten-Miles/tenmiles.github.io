---
layout:     post
title:      "Golang package 之 bufio"
subtitle:   ""
date:       2020-08-13 12:00:00
author:     "zhihao"
header-img: "img/post-bg-css.jpg"
tags:
    - golang
---
> 如果有工具能大幅度提高效率，那为什么不尝试一下呢。bufio包就是一个可以大幅度提高文件读写效率的包。

## bufio.Reader
通过它，我们可以从底层的 `io.Reader` 中更大批量的读取数据。这会使读取操作变少。如果数据读取时的块数量是固定合适的，底层媒体设备将会有更好的表现，也因此会提高程序的性能：
```
io.Reader --> buffer --> consumer
```
假设消费者想要从硬盘上读取10个字符(每次读取一个字符)。在底层实现上，这将会触发10次读取操作。如果硬盘按每个数据块四个字节来读取数据，那么 `bufio.Reader` 将会起到帮助作用。底层引擎将会缓存整个数据块，然后提供一个可以挨个读取字节的 API 给消费者：
```
abcd -----> abcd -----> a
            abcd -----> b
            abcd -----> c
            abcd -----> d
efgh -----> efgh -----> e
            efgh -----> f
            efgh -----> g
            efgh -----> h
ijkl -----> ijkl -----> i
            ijkl -----> j
```
`----->`代表读取操作,这个方法仅需要从硬盘读取三次，而不是10次。

#### Reset
通过使用 `Reset`,缓存也可以被复用,我们可以避免冗余的内存分配和不必要的垃圾回收工作。
```golang
s1 := strings.NewReader("abcd")
r := bufio.NewReader(s1)
b := make([]byte, 3)
_, err := r.Read(b)
if err != nil {
    panic(err)
}
fmt.Printf("%q\n", b)
s2 := strings.NewReader("efgh")
r.Reset(s2)
_, err = r.Read(b)
if err != nil {
    panic(err)
}
fmt.Printf("%q\n", b)
"abc"
"efg"
```

#### Discard
这个方法将会丢弃 n 个字节的，返回时也不会返回被丢弃的 n 个字节。如果 `bufio.Reader` 缓存了超过或者等于 n 个字节的数据。那么其将不必从 `io.Reader` 中读取任何数据。其只是简单的从缓存中略去前 n 个字节：
```golang
type R struct{}
func (r *R) Read(p []byte) (n int, err error) {
    fmt.Println("Read")
    copy(p, "abcdefghijklmnop")
    return 16, nil
}
func main() {
    r := new(R)
    br := bufio.NewReaderSize(r, 16)
    buf := make([]byte, 4)
    br.Read(buf)
    fmt.Printf("%q\n", buf)
    br.Discard(4)
    br.Read(buf)
    fmt.Printf("%q\n", buf)
}
Read
"abcd"
"ijkl"
```
调用 `Discard` 方法将不会从 `reader` `r` 中读取数据。另一种情况，缓存中数据量小于 n，那么 `bufio.Reader` 将会读取需要数量的数据来确保被丢弃的数据量不会少于 n：
```golang
type R struct{}
func (r *R) Read(p []byte) (n int, err error) {
    fmt.Println("Read")
    copy(p, "abcdefghijklmnop")
    return 16, nil
}
func main() {
    r := new(R)
    br := bufio.NewReaderSize(r, 16)
    buf := make([]byte, 4)
    br.Read(buf)
    fmt.Printf("%q\n", buf)
    br.Discard(13)
    fmt.Println("Discard")
    br.Read(buf)
    fmt.Printf("%q\n", buf)
}
Read
"abcd"
Read
Discard
"bcde"
```
由于调用了 `Discard` 方法，所以读取方法被调用了两次。

#### Read
`Read` 方法是 `bufio.Reader` 的核心。它和 [io.Reader](https://golang.org/pkg/io/#Reader) 的唯一方法具有相同的签名。因此 `bufio.Reader` 实现了这个普遍存在的接口：
```
type Reader interface {
    Read(p []byte) (n int, err error)
}
```
`bufio.Reader` 的 `Read` 方法从底层的 `io.Reader` 中一次读取最大的数量:

如果内部缓存具有至少一个字节的数据，那么无论传入的切片的大小(len(p))是多少，`Read` 方法都将仅仅从内部缓存中获取数据，不会从底层的 `reader` 中读取任何数据:
```golang
func (r *R) Read(p []byte) (n int, err error) {
    fmt.Println("Read")
    copy(p, "abcd")
    return 4, nil
}
func main() {
    r := new(R)
    br := bufio.NewReader(r)
    buf := make([]byte, 2)
    n, err := br.Read(buf)
    if err != nil {
        panic(err)
    }
    buf = make([]byte, 4)
    n, err = br.Read(buf)
    if err != nil {
        panic(err)
    }
    fmt.Printf("read = %q, n = %d\n", buf[:n], n)
}
Read
read = "cd", n = 2
```
我们的 `io.Reader` 实例无线返回「abcd」(不会返回 `io.EOF`)。 第二次调用 `Read`并传入长度为4的切片，但是内部缓存在第一次从 `io.Reader` 中读取数据之后已经具有数据「cd」，所以 `bufio.Reader` 返回缓存中的数据数据，而不和底层 `reader` 进行通信。

2.如果内部缓存是空的，那么将会执行一次从底层 `io.Reader` 的读取操作。 从前面的例子中我们可以清晰的看到如果我们开启了一个空的缓存，然后调用:
```
n, err := br.Read(buf)
```
将会触发读取操作来填充缓存。

3.如果内部缓存是空的，但是传入的切片长度大于缓存长度，那么 `bufio.Reader` 将会跳过缓存，直接读取传入切片长度的数据到切片中:
```golang
type R struct{}
func (r *R) Read(p []byte) (n int, err error) {
    fmt.Println("Read")
    copy(p, strings.Repeat("a", len(p)))
    return len(p), nil
}
func main() {
    r := new(R)
    br := bufio.NewReaderSize(r, 16)
    buf := make([]byte, 17)
    n, err := br.Read(buf)
    if err != nil {
        panic(err)
    }
    fmt.Printf("read = %q, n = %d\n", buf[:n], n)
    fmt.Printf("buffered = %d\n", br.Buffered())
}
Read
read = "aaaaaaaaaaaaaaaaa", n = 17
buffered = 0
```
从 `bufio.Reader` 读取之后，内部缓存中没有任何数据(buffered = 0)

## bufio.Writer
多次进行小量的写操作会影响程序性能。每一次写操作最终都会体现为系统层调用，频繁进行该操作将有可能对 CPU 造成伤害。而且很多硬件设备更适合处理块对齐的数据，例如硬盘。为了减少进行多次写操作所需的开支，golang 提供了 `bufio.Writer`。数据将不再直接写入目的地(实现了 `io.Writer` 接口)，而是先写入缓存，当缓存写满后再统一写入目的地：
```
producer --> buffer --> io.Writer
```
下面具体看一下在9次写入操作中(每次写入一个字符)具有4个字符空间的缓存是如何工作的：
```
producer        buffer       destination (io.Writer)
   a    ----->    a
   b    ----->    ab
   c    ----->    abc
   d    ----->    abcd
   e    ----->    e      ----->   abcd
   f    ----->    ef
   g    ----->    efg
   h    ----->    efgh
   i    ----->    i      ----->   abcdefgh
```
`----->` 箭头代表写入操作
`bufio.Writer` 底层使用 []byte 进行缓存
```golang
type Writer int
func (*Writer) Write(p []byte) (n int, err error) {
    fmt.Println(len(p))
    return len(p), nil
}
func main() {
    fmt.Println("Unbuffered I/O")
    w := new(Writer)
    w.Write([]byte{'a'})
    w.Write([]byte{'b'})
    w.Write([]byte{'c'})
    w.Write([]byte{'d'})
    fmt.Println("Buffered I/O")
    bw := bufio.NewWriterSize(w, 3)
    bw.Write([]byte{'a'})
    bw.Write([]byte{'b'})
    bw.Write([]byte{'c'})
    bw.Write([]byte{'d'})
    err := bw.Flush()
    if err != nil {
        panic(err)
    }
}
Unbuffered I/O
1
1
1
1
Buffered I/O
3
1
```

没有被缓存的 I/O：意味着每一次写操作都将直接写入目的地。我们进行4次写操作，每次写操作都映射为对 Write 的调用，调用时传入的参数为一个长度为1的 byte 切片。

使用了缓存的 I/O：我们使用三个字节长度的缓存来存储数据，当缓存满时进行一次 flush 操作(将缓存中的数据进行处理)。前三次写入写满了缓存。第四次写入时检测到缓存没有剩余空间，所以将缓存中的积累的数据写出。字母 d 被存储了，但在此之前 Flush 被调用以腾出空间。当缓存被写到末尾时，缓存中未被处理的数据需要被处理。`bufio.Writer` 仅在缓存充满或者显式调用 Flush 方法时处理(发送)数据。

> `bufio.Writer` 默认使用 4096 长度字节的缓存，可以使用 NewWriterSize 方法来设定该值

#### Reset
缓存是 bufio 的核心部分。通过使用 Reset 方法，Writer 可以用于不同的目的对象。重复使用 Writer 缓存减少了内存的分配。而且减少了额外的垃圾回收工作：
```golang
type Writer1 int
func (*Writer1) Write(p []byte) (n int, err error) {
    fmt.Printf("writer#1: %q\n", p)
    return len(p), nil
}
type Writer2 int
func (*Writer2) Write(p []byte) (n int, err error) {
    fmt.Printf("writer#2: %q\n", p)
    return len(p), nil
}
func main() {
    w1 := new(Writer1)
    bw := bufio.NewWriterSize(w1, 2)
    bw.Write([]byte("ab"))
    bw.Write([]byte("cd"))
    w2 := new(Writer2)
    bw.Reset(w2)
    bw.Write([]byte("ef"))
    bw.Flush()
}
writer#1: "ab"
writer#2: "ef"
```
这段代码中有一个 bug。在调用 Reset 方法之前，我们应该使用 Flush flush缓存。 由于 Reset 只是简单的丢弃未被处理的数据，所以已经被写入的数据 cd 丢失了：
```golang
func (b *Writer) Reset(w io.Writer) {
    b.err = nil
    b.n = 0
    b.wr = w
}
```