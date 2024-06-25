# 1

# 单例模式



此篇整理了几种常见的单例模式代码示例，再有面试官让手撕单例模式，便能心中有码，下笔有神。

### 为什么要有单例模式
实际编程应用场景中，有一些对象其实我们只需要一个，比如线程池对象、缓存、系统全局配置对象等。这样可以就保证一个在全局使用的类不被频繁地创建与销毁，节省系统资源。
### 实现单例模式的几个要点
1. 首先要确保全局只有一个类的实例。  
要保证这一点，至少类的构造器要私有化。  
2. 单例的类只能自己创建自己的实例。  
因为，构造器私有了，但是还要有一个实例，只能自己创建咯！    
3. 单例类必须能够提供自己的唯一实例给其他类  
就是要有一个公共的方法能返回该单例类的唯一实例。
### 单例模式的6种实现
#### 1、饿汉式—静态常量方式（线程安全）
```java
public class Singleton {  
    private static Singleton instance = new Singleton();  
    private Singleton (){}  
    public static Singleton getInstance() {  
    return instance;  
    }  
}
```
类加载时就初始化实例，避免了多线程同步问题。天然线程安全。
#### 2、饿汉式—静态代码块方式（线程安全）
其实就是在上面 静态常量饿汉式 实现上稍微变动了一下，将类的实例化放在了静态代码块中而已。其他没区别。
```java
public class Singleton {
    private static Singleton instance;
    static {
        instance = new Singleton();
    }
    private Singleton() {}
    public static Singleton getInstance() {
        return instance;
    }
}
```
#### 3、懒汉式（线程不安全）
```java
public class Singleton {
    private static Singleton singleton;
    private Singleton() {}
    public static Singleton getInstance() {
        if (singleton == null) {
            singleton = new Singleton();
        }
        return singleton;
    }
}
```
这是最基本的实现方式，第一次调用才初始化，实现了懒加载的特性。多线程场景下禁止使用，因为可能会产生多个对象，不再是单例。
#### 4、懒汉式（线程安全，方法上加同步锁）
```java
public class Singleton {
    private static Singleton singleton;
    private Singleton() {}
    public static synchronized Singleton getInstance() {
        if (singleton == null) {
            singleton = new Singleton();
        }
        return singleton;
    }
}
```
和上面 懒汉式（线程不安全）实现上唯一不同是：获取实例的getInstance()方法上加了同步锁。保证了多线程场景下的单例。但是效率会有所折损，不过还好。
#### 5、双重校验锁（线程安全，效率高）
```java
public class Singleton {
	private volatile static Singleton singleton;
	private Singleton() {}
	public static Singleton getSingleton() {
		if (singleton == null) {
			synchronized (Singleton.class) {
				if (singleton == null) {
						singleton = new Singleton();
				}
			}
		}
		return singleton;
	}
}
```
此种实现中不用每次需要获得锁，减少了获取锁和等待的事件。  
注意volatile关键字的使用，保证了各线程对singleton静态实例域修改的可见性。
#### 6、静态内部类实现单例（线程安全、效率高）
```java
public class Singleton {  
    private static class SingletonHolder {  
    private static final Singleton INSTANCE = new Singleton();  
    }  
    private Singleton (){}  
    public static final Singleton getInstance() {  
    return SingletonHolder.INSTANCE;  
    }  
}
```
这种方式下 Singleton 类被装载了，instance 不一定被初始化。因为 SingletonHolder 类没有被主动使用，只有通过显式调用 getInstance 方法时，才会显式装载 SingletonHolder 类，从而实例化 instance。  
注意内部类SingletonHolder要用static修饰且其中的静态变量INSTANCE必须是final的。

此篇完。单例模式掌握至此，足以应付“手写单例”的面试场景。



# 原型模式

### 什么是原型模式

什么是原型模式，就是根据一个已经存在的对象实例，复制创建出多个对象实例的设计方法。**已经存在的对象实例**就是原型对象。原型模式属于创建型的设计模式。

当创建对象的代价交高时，可是使用原型模式复制拷贝对象，这样更做效率更高。

原型模式复制对象一般会用到Object类的clone方法。在Java中实现对象拷贝或克隆，使用clone()方法。

能够实现克隆的Java类必须实现一个标识接口Cloneable，表示这个Java类支持复制。如果一个类没有实现这个接口但是调用了clone()方法，Java编译器将抛出一个CloneNotSupportedException异常。

### 原型模式的几种使用场景

1、通过 new 产生一个对象需要非常繁琐的数据准备或访问权限，则可以使用原型模式，创建原型对象后缓存，再使用该种对象时，返回它的拷贝。

2、一个对象需要提供给其他对象访问，而且各个调用者可能都需要修改其值时，可以考虑使用原型模式拷贝多个对象供调用者使用。 

3、如果想要让生成实例的框架不再依赖于具体的类，这时，不能指定类名来生成实例，而要事先“注册”一个“原型”实例，然后通过复制该实例来生成新的实例。

### 原型模式代码实现

将创建一个抽象类 Animal和扩展了 Animal类的实体类Cat、Dog。下一步是定义类 AnimalCache，该类把 Animal 对象存储在一个 Hashtable 中，并在请求的时候返回它们的克隆。

1、先定义一个原型抽象类，可以是接口，实现Cloneable接口。

```java
public abstract class Animal implements Cloneable{

    // 名字
    protected String name;

    // 重量
    protected float weight;

    // 叫声，定义为抽象的
    abstract void bark();

    // 克隆，此处直接使用object的clone来进行对象拷贝
    public Object clone(){
        Object clone = null;
        try {
            clone = super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return clone;
    }
}
```

2、创建两个具体的实体类，继承或实现上面的原型抽象类或接口。

```java
public class Cat extends Animal{

    public Cat(String name, float weight){
        this.name = name;
        this.weight = weight;
    }
    @Override
    void bark() {
        System.out.println("猫叫。。。。。");
    }
}

public class Dog extends Animal{

    public Dog(String name, float weight){
        this.name = name;
        this.weight = weight;
    }
    @Override
    void bark() {
        System.out.println("狗叫。。。。。");
    }
}
```

3、创建一个类，从数据库获取实体类，并把它们存储在一个 Hashtable 中。

```java
public class AnimalCache {
    private static Hashtable<String, Animal> animalMap
            = new Hashtable<String, Animal>();

    public static Animal getAnimal(String name) {
        Animal cachedAnimal = animalMap.get(name);
        return (Animal) cachedAnimal.clone();
    }

    public static void loadCache() {
        Cat cat = new Cat("tom",50);
        animalMap.put("cat",cat);

        Dog dog = new Dog("小六",100);
        animalMap.put("dog",dog);

    }
}
```

4、创建一个PrototypePatternTest 使用 AnimalCache 类来获取存储在 Hashtable 中的形状的克隆。

```java
public class PrototypePatternTest {
    public static void main(String[] args) {
        AnimalCache.loadCache();
        Animal dog1 = AnimalCache.getAnimal("dog");
        Animal dog2 = AnimalCache.getAnimal("dog");
        System.out.println(dog1);
        System.out.println(dog2);
        dog2.bark();
        Animal cat2 = AnimalCache.getAnimal("cat");
        cat2.bark();
    }
}
```

执行程序，输出结果如下：

```java
com.herp.pattern.prototype.Dog@4554617c
com.herp.pattern.prototype.Dog@74a14482
狗叫。。。。。
猫叫。。。。。
```

可以看到两个，获取了两次dog对象，每次获取到的都是不同的对象。同时也是原型对象的拷贝。



# 代理模式

代理模式就是为一个对象提供一个代理对象，由这个代理对象控制对该对象的访问。

理解代理模式，可以对照生活中的一些具体例子，比如房产中介、二手车交易市场、经纪人等。

### 为什么要用代理模式

通过使用代理模式，我们避免了直接访问目标对象时可能带来的一些问题，比如：远程调用，需要使用远程代理来帮我们处理一些网络传输相关的细节逻辑；可能需要基于某种权限控制对目标资源的访问，可以使用保护代理等。

总的来说，通过是用代理模式，我们可以控制对目标对象的访问，可以在真实方法被调用前或调用后，通过代理对象加入额外的处理逻辑。

### 代理模式分类

代理模式分为静态代理和动态代理。动态代理根据实现不同又可细分为JDK动态代理和cglib动态代理。

**静态代理**是由程序员创建或工具生成代理类的源码，再编译代理类。所谓静态也就是在程序运行前就已经存在代理类的字节码文件，代理类和委托类的关系在运行前就确定了。

**动态代理**是在实现阶段不用关心代理类，而在运行时动态生成代理类的。

### 静态代理

以房哥买房子为例，用代码实现静态代理。
1、首先建立一个Seller接口

```java
public interface Seller {
    void sell();
}
```

2、创建实现类，房哥，有一个方法，就是买房子

```java
public class FangGe implements Seller{
    @Override
    public void sell() {
        System.out.println("房哥要出手一套四合院");
    }
}
```

3、买房子需要找到买家，达成交易后还要办理过户等其他手续，房哥只想卖房收钱就完了。因此，需要找一个代理来帮房哥处理这些杂事。

我们创建一个代理类FangGeProxy，代理类也需要实现Seller接口，行为上要保持和FangGe一样，都是要卖房子。同时该代理类还需要持有房哥的引用。

```java
public class FangGeProxy implements Seller{
    private FangGe fangGe;

    public FangGeProxy(FangGe fangGe){
        this.fangGe = fangGe;
    }
    @Override
    public void sell() {
        findBuyer();
        fangGe.sell();
        afterSell();
    }
    
    public void findBuyer(){
        System.out.println("代理帮助寻找买主");
    }
    
    public void afterSell(){
        System.out.println("达成交易后，办理相关手续");
    }
}
```

可以看到，房哥的代理类通过findBuyer()和afterSell()两个方法帮助房哥完成了其他一些杂事。

4、测试类

```java
public class StaticProxyTest {
    public static void main(String[] args) {
        Seller seller = new FangGeProxy(new FangGe());
        seller.sell();
    }
}
```

输出：

```java
代理帮助寻找买主
房哥要出手一套四合院
达成交易后，办理相关手续
```

最后，看下类图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191125214537361.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTY1MDE4,size_16,color_FFFFFF,t_70)
**静态代理的问题：**
1、由于静态代理类在编译前已经确定了代理的对象，因此静态代理只能代理一种类型的类，如果要给大量的类做代理，就需要编写大量的代理类；

2、如果我们要给Seller，也就是目标对象要增加一些方法，则需要同步修改代理类，不符合开闭原则。

### JDK动态代理

JDK的动态代理依赖于jdk给我们提供的类库实现，是一种基于接口实现的动态代理，在编译时并不知道要代理哪个类，而是在运行时动态生成代理类。同时也解决了静态代理中存在的问题。

我们接上上面静态代理的例子，继续实现JDK的动态代理。
1、我们建一个方法转发的处理器类，该类需要实现InvocationHandler接口。

```java
public class SellerInvocationHandler implements InvocationHandler {

    // 要代理的真实对象
    private Object target;

    /**
     * 使用Proxy类静态方法获取代理类实例
     */
    public Object getProxyInstance(Object target){
        this.target = target;
        Class<?> clazz = target.getClass();
        return Proxy.newProxyInstance(clazz.getClassLoader(),clazz.getInterfaces(),this);
    }
    
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        before();
        Object obj = method.invoke(this.target, args);
        after();
        return obj;
    }

    private void before() {
        System.out.println("执行方法前");
    }
    
    private void after() {
        System.out.println("执行方法后");
    }
}
```

2、新建JDK动态代理测试类，首先代理房哥卖房子

```java
public class JDKDynamicProxyTest {
    public static void main(String[] args) {

        // new一个房哥，下面帮房哥找个代理
        FangGe fangGe = new FangGe();
        SellerInvocationHandler sellerInvocationHandler = new SellerInvocationHandler();
        
        // 房哥的代理对象
        Seller seller = (Seller) sellerInvocationHandler.getProxyInstance(fangGe);
        seller.sell();

    }
}
```

输出：

```java
执行方法前
房哥要出手一套四合院
执行方法后
```

可以看到，完成了代理。
3、接下来我们新建另外一个类，User类，并使用JDK动态代理完成代理User类

```java
public interface IUser {
    void sayHello();

    void work();
}

public class UserImpl implements IUser{
    @Override
    public void sayHello() {
        System.out.println("hello,我是小明");
    }

    @Override
    public void work() {
        System.out.println("我正在写代码");
    }
}
```

修改测试类，

```java
public class JDKDynamicProxyTest {
    public static void main(String[] args) {

/*        // new一个房哥，下面帮房哥找个代理
        FangGe fangGe = new FangGe();
        SellerInvocationHandler sellerInvocationHandler = new SellerInvocationHandler();

        // 房哥的代理对象
        Seller seller = (Seller) sellerInvocationHandler.getProxyInstance(fangGe);
        seller.sell();*/

        // 代理user类
        IUser user = new UserImpl();
        SellerInvocationHandler sellerInvocationHandler = new SellerInvocationHandler();
        IUser userProxy = (IUser) sellerInvocationHandler.getProxyInstance(user);
        userProxy.sayHello();
        userProxy.work();

    }
}
```

输出：

```java
执行方法前
hello,我是小明
执行方法后
执行方法前
我正在写代码
执行方法后

```

可以看到，我们SellerInvocationHandler 并未做任何改动，它便能为UserImpl类生成代理，并在执行方法的前后增加额外的执行逻辑。

### cglib动态代理

JDK动态代理有一个局限就是，被代理的类必须要实现接口。如果被代理的类没有实现接口，则JDK动态代理就无能为力了。这个时候该cglib动态代理上场了。

> CGLIB是一个功能强大，高性能的代码生成包。它为没有实现接口的类提供代理，为JDK的动态代理提供了很好的补充。通常可以使用Java的动态代理创建代理，但当要代理的类没有实现接口或者为了更好的性能，CGLIB是一个好的选择。

1、新建一个MyCglibInterceptor，实现MethodInterceptor接口。该类类似于JDK动态代理中的InvocationHandler实例，是实现cglib动态代理的主要类。

```java
public class MyCglibInterceptor implements MethodInterceptor {

    public Object getCglibProxyInstance(Object object){
        // 相当于Proxy，创建代理的工具类
        Enhancer enhancer = new Enhancer();
        enhancer.setSuperclass(object.getClass());
        enhancer.setCallback(this);
        return enhancer.create();
    }

    public Object intercept(Object o, Method method, Object[] objects, MethodProxy methodProxy) throws Throwable {
        before();
        Object obj = methodProxy.invokeSuper(o, objects);
        after();
        return obj;
    }

    private void before() {
        System.out.println("执行方法之前");
    }

    private void after() {
        System.out.println("执行方法之后");
    }
}
```

2、新建cglib动态代理的测试类，先代理上面例子中的User类。

```java
public class CglibDynamicProxyTest {
    public static void main(String[] args) {
        MyCglibInterceptor myCglibInterceptor = new MyCglibInterceptor();
        IUser userCglibProxy = (IUser) myCglibInterceptor.getCglibProxyInstance(new UserImpl());
        userCglibProxy.sayHello();
        userCglibProxy.work();
    }
}
```

输出：

```java
执行方法之前
hello,我是小明
执行方法之后
执行方法之前
我正在写代码
执行方法之后
```

3、新建一个类HelloWorld，不实现任何接口，为该类实现动态代理。

```java
public class HelloWorld {
    public void hello(){
        System.out.println("世界这么大，我想去看看");
    }
}
```

测试代理类

```java
public class CglibDynamicProxyTest {
    public static void main(String[] args) {
/*        MyCglibInterceptor myCglibInterceptor = new MyCglibInterceptor();
        IUser userCglibProxy = (IUser) myCglibInterceptor.getCglibProxyInstance(new UserImpl());
        userCglibProxy.sayHello();
        userCglibProxy.work();*/

        // 代理未实现任何接口的类
        MyCglibInterceptor myCglibInterceptor = new MyCglibInterceptor();
        HelloWorld helloWorldProxy = (HelloWorld) myCglibInterceptor.getCglibProxyInstance(new HelloWorld());
        helloWorldProxy.hello();
    }
}
```

输出：

```java
执行方法之前
世界这么大，我想去看看
执行方法之后
```

使用cglib动态代理，我们实现了对普通类的代理。
