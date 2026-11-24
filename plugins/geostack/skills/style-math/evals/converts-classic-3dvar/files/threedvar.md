# 3D-Var

The analysis $\bm{x}^a$ minimises the cost function

$$J(\bm{x}) = \frac{1}{2}(\bm{x}-\bm{x}_b)^T \bm{B}^{-1} (\bm{x}-\bm{x}_b) + \frac{1}{2}(\bm{y}^o - H(\bm{x}))^T \bm{R}^{-1} (\bm{y}^o - H(\bm{x}))$$

For a linear observation operator $\bm{H}$ the minimiser is

$$\bm{x}^a = \bm{x}_b + \bm{K}(\bm{y}^o - \bm{H}\bm{x}_b), \qquad \bm{K} = \bm{B}\bm{H}^T(\bm{H}\bm{B}\bm{H}^T + \bm{R})^{-1}$$

and the analysis error covariance is $\bm{A} = (\bm{I} - \bm{K}\bm{H})\bm{B}$. The optimum is $\bm{x}^* = \bm{x}^a$.
