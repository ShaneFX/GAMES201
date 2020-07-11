使用Taichi 实现了一个基础的三维烟雾模拟器，基本是按照第四课的内容，包括了Advection 和 Projection。

* Advection
   代码中的advection 有Semi-Lagrangian 和 BFECC，但是BFECC的Clipping部分作用于向量时会出错，目前还未解决，所以我对速度场使用了Semi-Lagrangian方法，对density场使用了BFECC。

* Projection
  Projection部分主要参考了案例 stable_fluid.py， mgpcg.py和mgpcg_advanced.py。
我代码中的线性系统求解方法为 Jacobi iteration， red-black Gauss-Seidel， Conjugate gradients, Multigrid preconditioned conjugate gradients。 前两个比较简单，后两个主要参考了案例的写法，还得加深理解。
  其中Jacobi iteration 在迭代次数少的时候（如下图，5次迭代），烟雾形态比较模糊。而Gauss-Seidel在同样迭代次数少的情况下更准确。
<img width="500px" src="https://github.com/ShaneFX/GAMES201/blob/master/HW01/images/01.jpg">
  Conjugate gradients方法在不用 preconditioner 的情况下收敛很慢，所以迭代次数少的时候形态会有较大差异。使用Jacobi preconditioner 的情况简单试了一下，没有太大差异。而使用Multigrid preconditioner 的效果却很惊人，在迭代十多次的时候就已经收敛了，而且就算只用5个iteration，最后的结果也非常好，和15次的几乎一样。
<img width="500px" src="https://github.com/ShaneFX/GAMES201/blob/master/HW01/images/2.jpg">

* 其它
  我使用的dt 为0.04， 测试中发现大于0.07左右就会有Artifact 产生，当然这与初始发射源和速度的设置是有关系的。
