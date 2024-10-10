# 1.简介

Spring是一个轻量级的企业级的Java开发框架。主要是用来替代原来更加重量级的企业级Java技术，比如EJB(Enterprise JavaBean)、Java数据对象（Java Data Object）等。Spring的出现极大简化了Java开发。

另外Spring框架是一个一体化的框架，它不仅能无缝对接比如Struts、Hibernate等传统框架，也能很好地同其他各种企业级开发组件（比如Redis、MQ、Mybatis等）集成。

Spring发展到现在，已经不仅仅是一个开发框架了，而是一个生态。Spring框架本身提供了大量可集成到应用中组件，SpringBoot通过“约定优于配置的思想”进一步提高了开发效率，成为构建微服务应用的最佳选择，SpringCloud则提供了一套分布式工具组件，让构建分布式系统更加简单。

Spring就是要简化Java开发

Spring一直致力于简化Java开发使命中，为了降低Java开发的复杂性，Spring通过如下4种关键策略来简化Java开发：

 - 基于POJO的轻量级和最小侵入性编程；
 - 通过依赖注入和面向接口编程实现松耦合；
 - 基于切面和惯例进行声明式编程；
 - 通过切面和模板减少样板式代码。

## Spring框架中的几个重要概念

**依赖注入（DI）**

对象之间的依赖关系，不再由对象自身来维护了。而是由spring负责管理了。依赖关系将会由spring负责自动注入到需要的对象中。

**切面编程（AOP）**

应用中的一些横切关注点，比如日志、安全、事务管理等，各个模块都需要的服务，不应该耦合在各个业务模块中来。应该单独抽离出来成为一个切面，实现解耦。Spring提供了AOP的支持，可以通过配置实现切面编程。

**Ioc容器（ApplicationContext）**

在基于Spring的应用中，你的所有Bean对象都存在于Spring容器中，容器负责创建和管理他们的整个生命周期。ApplicationContext应用上下文对象是Spring容器的一种实现。通过应用上下文对象我们可以获取应用中bean。

## Spring模块概述

一图胜千言

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191204223212908.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTY1MDE4,size_16,color_FFFFFF,t_70)
可以看到spring其实包含了20多个不同的模块。

1、spring核心容器，包含四大模块，分别是Beans、Core、Context和SpEL。这是构成spring框架的核心组件。

2、再网上是AOP、Aspects切面编程相关组件，Instrunmentation是JVM添加代理，Messaging消息代理。

3、JDBC、ORM、JMS等数据访问组件。

4、Servlet、WebScoket等web层相关组件。

5、最后spring也提供了测试模块，可以集成Junit单元测试等。

### 推荐几个Spring学习途径

**1、 当然优先是spring官网，查看官方文档学习**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191204224235668.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTY1MDE4,size_16,color_FFFFFF,t_70)

**2、W3Cschool**

![https://www.w3cschool.cn/wkspring/](https://img-blog.csdnimg.cn/20191204224429520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTY1MDE4,size_16,color_FFFFFF,t_70)
**3、CSDN或博客园等技术文章**

Spring最基础的特性就是创建bean、管理bean之间的依赖关系。下面通过具体实例演示该如何装配我们应用中的bean。

# Spring主要的装配机制

- 在xml中进行显示的配置
- 在Java中进行显示的配置
- 隐式的bean发现机制和自动装配

三种装配方式可依据个人喜好选择使用，无限制。不过应尽可能地使用自动转配机制，因为可以少一大推的配置。在你必须要使用显示配置时，比如要装配一些第三方的bean对象是，可以使用显示的配置。推荐使用类型安全且比xml更加强大的JavaConfig。

自动装配bean

1、创建bean并添加@Component注解

spring通过面向接口编程实现松耦合，因此平时在开发时要养成先创建接口，然后再实现类的习惯。
此处先创建一个CentralProcessUnit接口

```java
public interface CentralProcessUnit {
    void compute();
}
```

然后创建一个具体的实现类HisiCentralProcessUnit：

```java
@Component
public class HisiCentralProcessUnit implements CentralProcessUnit{
    public void compute() {
        System.out.println("海思芯片，计算。。。。。");
    }
}
```

可以看到，在该类上有一个注解@Component，表明该类是一个组件，spring在扫描的时候会自动创建该类的对象。

当然了，只添加这个注解是不够的。

2、开启组件扫描功能

为了能完成spring 的自动装配功能，除了在需要创建bean的类上添加@Component注解之外，还需要开启组件扫描功能。spring的组件扫描功能默认是关闭的。

开启组件扫描有两种方式：

- JavaConfig配置方式
- xml配置方式

Java配置方式

```java
@Configuration
@ComponentScan
public class ScanConfig {
}
```

XML方式，在spring配置文件中增加如下这段就好：

```java
<context:component-scan base-package="com.herp"/>
```

3、新建测试类，验证bean是否注入

```java
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = ScanConfig.class)
public class AutoWiredBeanTest {

    @Autowired
    private CentralProcessUnit cpu;

    @Test
    public void testNotBeanNull(){
        Assert.assertNotNull(cpu);
    }
}
```

在该测试类中使用了@Autowired注解将扫描创建的bean对象注入到此测试类中。

通过上面的三个步骤我们可以看到，我们并没有使用new对象的方式，只是使用了一些spring的注解，spring便帮我们自动完成了对象的创建，并将其注入到了测试类中。

同样地，我们也可以在bean对象之间完成自动装配。我们再创建一个手机对象，然后将cpu芯片对象装配到手机对象中。

创建手机接口和实现类：

```java
@Component
public class HuaweiPhone implements IPhone{
    
    @Autowired
    private CentralProcessUnit cpu;
    
    public void compute() {
        cpu.compute();
    }
}
```

在HuaweiPhone 对象中自动装配CentralProcessUnit的实例。
**这里值得注意的是CentralProcessUnit是接口，如果有两个CentralProcessUnit实例，使用自动装配便会产生歧义性，因为spring容器不知道该装配哪个bean了。**

使用JavaConfig的显式配置

使用JavaConfig完成bean装配其实就是使用Java代码加spring的注解完成bean的配置、依赖的配置信息。

举例来说明：
本示例中，我们使用JavaConfig的方式完成CentralProcessUnit及IPhone实例的装配。

1、去掉HisiCentralProcessUnit 类上的@Component注解

```java
public class HisiCentralProcessUnit implements CentralProcessUnit {
    public void compute() {
        System.out.println("海思芯片，计算。。。。。");
    }
}
```

2、去掉ScanConfig中的@ComponentScan注解，只留下@Configuration注解，表示这是一个配置类。然后编写产生HisiCentralProcessUnit  bean的方法，并使用@Bean注解标注。

```java
@Configuration
public class ScanConfig {
    @Bean
    public CentralProcessUnit hisiCentralProcessUnit(){
        return new HisiCentralProcessUnit();
    }
}
```

3、如此便完成了最简单的bean的装配了。测试一下：

```java
public class JavaConfigWiredBeanTest {
    public static void main(String[] args) {
        ApplicationContext applicationContext = new AnnotationConfigApplicationContext(ScanConfig.class);
        HisiCentralProcessUnit cpu = applicationContext.getBean("hisiCentralProcessUnit", HisiCentralProcessUnit.class);
        cpu.compute();
    }
}
```

输出：

```java
海思芯片，计算。。。。。
```

可以看到完成了bean的转配。

4、同样地，我们去掉HuaweiPhone上的@Component注解，并提供一个带参数的构造器，注意参数类型是CentralProcessUnit（接口），而非具体实现类，具体类接下来我们会使用JavaConfig的配置的方式注入。

```java
public class HuaweiPhone implements IPhone {

    private CentralProcessUnit cpu;

    // 构造器注入
    public HuaweiPhone(CentralProcessUnit cpu){
        this.cpu = cpu;
    }

    public void compute() {
        System.out.println("华为手机+++");
        cpu.compute();
    }
}
```

5、我们在ScanConfig中加入huaweiPhone的配置，并注入hisiCentralProcessUnit实例。

```java
@Configuration
public class ScanConfig {

    @Bean
    public CentralProcessUnit hisiCentralProcessUnit(){
        return new HisiCentralProcessUnit();
    }

    @Bean
    public IPhone huaweiPhone(){
        return new HuaweiPhone(hisiCentralProcessUnit());
    }
}
```

6、验证依赖注入是否成功。使用AnnotationConfigApplicationContext容器类获取装配好的bean。

```java
public class JavaConfigWiredBeanTest {
    public static void main(String[] args) {
        ApplicationContext applicationContext = new AnnotationConfigApplicationContext(ScanConfig.class);
        // 依赖注入示例测试
        IPhone huaweiPhone = applicationContext.getBean("huaweiPhone",HuaweiPhone.class);
        huaweiPhone.compute();
    }
}
```

输出结果：

```java
华为手机+++
海思芯片，计算。。。。。
```

使用JavaConfig配置bean和依赖关系，具有极大的灵活性，我们只需要在带有@Bean注解的方法上最终产生一个bean实例即可，具体bean实例的产生逻辑只受Java语言自身的限制。**

比如我们可以这样配置一个bean：

```java
    @Bean
    public CentralProcessUnit randomCentralProcessUnit(){
        int num = (int) Math.floor(Math.random() * 2);
        if(num == 0){
            return new HisiCentralProcessUnit();
        }else {
            return new GaoTongCentralProcessUnit();
        }
    }
```

这个CentralProcessUnit 是随机生成HisiCentralProcessUnit芯片或者是GaoTongCentralProcessUnit芯片。

使用XML完成bean装配和DI

XML配置的方式是spring最早采用的配置bean、bean之间的依赖关系的方式。

使用XML配置其实很简单，使用<bean>元素声明一个bean即可。

**1、最简单的bean配置**

```java
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">

    <bean id="centralProcessUnit" class="com.herp.beanwired.xmlwired.GaoTongCentralProcessUnit"/>
</beans>
```

这样当spring发现<bean>元素时，会调用GaoTongCentralProcessUnit的默认构造器来创建bean。

**2、使用构造器依赖注入bean**

```java
    <bean id="iphone" class="com.herp.beanwired.xmlwired.HuaweiPhone">
        <constructor-arg ref="centralProcessUnit"/>
    </bean>

```

测试类

```java
public class XMLConfigWiredBeanTest {
    public static void main(String[] args) {
        ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext-beans.xml");
        IPhone phone = context.getBean("iphone",IPhone.class);
        phone.compute();
    }
}
```

输出：

```java
华为手机+++
高通芯片，计算。。。。
```

3、使用属性注入bean依赖

```java
    <bean id="centralProcessUnit_hw" class="com.herp.beanwired.xmlwired.HisiCentralProcessUnit"/>
    <bean id="anotherPhone" class="com.herp.beanwired.xmlwired.HuaweiPhone">
        <property name="centralProcessUnit" ref="centralProcessUnit_hw"/>
    </bean>
```

同时，需要在HuaweiPhone类中提供setter方法

```java
    public void setCentralProcessUnit(CentralProcessUnit centralProcessUnit){
        this.centralProcessUnit = centralProcessUnit;
    }
```

测试类，替换成获取id为anotherPhone 的bean对象：

```java
public class XMLConfigWiredBeanTest {
    public static void main(String[] args) {
        ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext-beans.xml");
        IPhone phone = context.getBean("anotherPhone",IPhone.class);
        phone.compute();
    }
}
```


输出如下：

```java
华为手机+++
海思芯片，计算。。。。。
```

总结

本节主要总结了spring装配bean和依赖注入的三种方式：组件扫描和自动装配、JavaConfig注解配置方式、XML配置方式。平时开发中优先使用组件扫描和spring自动发现的机制装配bean，这样可以省去大量配置类或配置文件，其次是使用JavaConfig的方式，一方面，它是类型安全的，另外在配置bean时提供了更大的灵活性。最后选用XML配置文件的方式装配bean。



### 根据不同的环境来装配不同的bean

企业级开发中，我们一般有多种环境，比如开发环境、测试环境、UAT环境和生产环境。而系统中有些配置是和环境强相关的，比如数据库相关的配置，与其他外部系统的集成等。

如何才能实现一个部署包适用于多种环境呢？

Spring给我们提供了一种解决方案，这便是条件化装配bean的机制。最重要的是这种机制是在运行时决定该注入适用于哪个环境的bean对象，不需要重新编译构建。

下面使用Spring的profile机制实现dataSource对象的条件化装配。

**1、给出开发环境、测试环境、生产环境dataSource的不同实现类**

说明：此处只为演示条件化装配bean，不做真实数据源对象模拟。

```java
public interface DataSource {
    void show();
}

public class DevDataSource implements DataSource{
    
    public DevDataSource(){
        show();
    }
    public void show() {
        System.out.println("开发环境数据源对象");
    }
}

public class TestDataSource implements DataSource{

    public TestDataSource() {
        show();
    }

    public void show() {
        System.out.println("测试环境数据源对象");
    }
}

public class ProDataSource implements DataSource{

    public ProDataSource() {
        show();
    }

    public void show() {
        System.out.println("生产环境数据源对象");
    }
}
```

**2、使用profile配置条件化bean**

其实profile的原理就是将不同的bean定义绑定到一个或多个profile之中，在将应用部署到不同的环境时，确保对应的profile处于激活状态即可。

这里我们使用JavaConfig的方式配置profile  bean

```java
@Configuration
public class DataSourceConfig {
    
    @Bean
    @Profile("dev")
    public DataSource devDataSource(){
        return new DevDataSource();
    }

    @Bean
    @Profile("test")
    public DataSource testDataSource(){
        return new TestDataSource();
    }

    @Bean
    @Profile("pro")
    public DataSource proDataSource(){
        return new ProDataSource();
    }
}
```

可以看到我们使用了@Profile注解，将不同环境的bean绑定到了不同的profile中。

**3、激活profile**

只要上面的两步还不行，我们还必须激活profile，这样Spring会依据激活的哪个profile，来创建并装配对应的bean对象。

激活profile需要两个属性。

```java
spring.profiles.active
spring.profiles.default
```

可以在web.xml中配置Web应用的上下文参数，来激活profile属性。比如在web.xml中增加如下配置来激活dev的profile：

```java
    <context-param>
        <param-name>spring.profiles.active</param-name>
        <param-value>dev</param-value>
    </context-param>
```

**4、测试条件化装配**

使用@ActiveProfiles注解在测试类中激活指定profile。

```java
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = {DataSourceConfig.class})
@ActiveProfiles("dev")
public class TestConditionDataSource {

    @Autowired
    private DataSource dataSource;

    @Test
    public void testDataSource(){
        Assert.assertNotNull(dataSource);
    }

}
```

输出：

```java
开发环境数据源对象
```

我们profile换成生产环境的pro试下，

```java
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = {DataSourceConfig.class})
@ActiveProfiles("pro")
public class TestConditionDataSource {

    @Autowired
    private DataSource dataSource;

    @Test
    public void testDataSource(){
        Assert.assertNotNull(dataSource);
    }

}
```

输出：

```java
生产环境数据源对象
```

通过spring的profile机制，我们实现了不同环境dataSource数据源对象的条件化装配。比较简单，就两步：1、使用@Profile注解为不同的bean配置profile（当然这里也可以是xml的方式），2、根据不同环境激活不同的profile。

使用@Conditional注解实现条件化的bean

Spring 4.0引入的新注解@Conditional注解，它可以用到带有@Bean注解的方法上，如果给定的条件计算结果为true，就会创建这个bean，否则不创建。

**1、我们创建一个helloWorld对象**

```java
public class HelloWorld {

    public void sayHello(){
        System.out.println("conditional 装配helloworld");
    }
}
```

**2、创建配置类**

在该配置类中我们首先使用了@PropertySource注解加载了属性文件hello.properties，其次可以看到在helloWorld的bean配置中，除了@Bean注解外，多了一个@Conditional注解，不错，@Conditional注解是我们实现条件化装配bean的核心注解。

@Conditional注解中有一个HelloWorldConditional类，该类定义了我们创建该bean对象的条件。


```java
@Configuration
@PropertySource("classpath:hello.properties")
public class HelloWorldConfig {

    @Bean
    @Conditional(HelloWorldConditional.class)
    public HelloWorld helloWorld(){
        return new HelloWorld();
    }
}
```

**3、创建条件类HelloWorldConditional，需要实现Condition接口。**

实现了Condition接口，重写了matches方法，在该方法中我们检测了环境变量中是否有hello属性，如果有就创建。没有则忽略。

**注意：hello.properties中属性会存储到spring的Environment对象中，因此我们可以检测到其中的属性是否存在。**


```java
public class HelloWorldConditional implements Condition {
    public boolean matches(ConditionContext conditionContext, AnnotatedTypeMetadata annotatedTypeMetadata) {
        return conditionContext.getEnvironment().containsProperty("hello");
    }
}
```

4、测试条件装配

```java
public class HelloWorldConditionTest {
    public static void main(String[] args) {
        ApplicationContext applicationContext = new AnnotationConfigApplicationContext(HelloWorldConfig.class);
        HelloWorld helloWorld = applicationContext.getBean("helloWorld",HelloWorld.class);
        helloWorld.sayHello();
    }
}
```

开始，我们在hello.properties中增加一条属性，

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208172943978.png)

运行测试示例，会输出：

```java
conditional 装配helloworld
```

说明此时，bean已成功装配。

如果我们注释掉hello.properties的这行属性。再次运行示例，则会提示bean不存在。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208173201111.png)
提示没有“helloWorld”的bean对象，说明了条件不满足不会创建bean对象。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208173225370.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzM3OTY1MDE4,size_16,color_FFFFFF,t_70)

### 总结

Spring条件化装配bean的两种方式，第一种是使用profile机制，在bean的配置类中使用@profile注解，标识哪些bean对应哪个profile配置，然后在web.xml或Servlet启动参数中配置激活哪个profile来实现条件装配；第二种是使用@Conditional注解，在带有@Bean注解的方法上增加@Conditional注解，在注解属性值中提供一个实现了Condition接口的类（该类会重写matches方法，定义具体的创建条件）。<完>

<section class="_editor"><section class="_editor"><p style="text-align:center"><img src="http://img.96weixin.com/ueditor/20191122/1574428873195184.png" alt="1574428873195184.png" _src="http://img.96weixin.com/ueditor/20191122/1574428873195184.png"></p></section></section>

