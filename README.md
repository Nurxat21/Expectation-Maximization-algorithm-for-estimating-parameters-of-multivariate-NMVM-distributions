Given the data set \( X = \{x_1, \ldots, x_n\} \) where \( X \in \mathbb{R}^{d \times n} \), the objective is to fit this random variable using multivariate generalized hyperbolic distributions. The parameters associated with this distribution are represented as \( \xi = (\lambda, \chi, \psi, \Sigma, \mu, \gamma) \). The log-likelihood function that requires maximization is expressed as follows:

$$
\log L\left({\xi} ; {x}_1, \cdots, {x}_n\right)=\sum_{i=1}^n \log f_{{x}_{{i}}}\left({x}_i ; {\xi}\right).
$$

It is hard to obtain the optimal solution of the above objective function directly if the data dimension is more than three. The main idea of the EM algorithm is to optimize the following augmented log-likelihood function:

$$
\log \tilde{L}\left({\xi} ; {x}_1, \cdots, {x}_n, w_1, \cdots, w_n\right)=\sum_{i=1}^n \log f_{{x}_{{i}}, W_i}\left({x}_{{i}}, w_i ; {\xi}\right)
$$

where the latent mixing variables \(w_1, \cdots, w_n\) are not observable at the beginning. 

---

### Mean-variance mixture representation

By the mean-variance mixture definition of generalized hyperbolic distributions, the log-likelihood function can be rewritten as

$$
\begin{aligned}
\log \tilde{L}\left({\xi}\right)
&= \sum_{i=1}^n \log f_{{x}_i \mid w_i}\left({x}_i \mid w_i ; {\mu}, \Sigma, {\gamma}\right)+ \sum_{i=1}^n \log h_{w_i}\left(w_i ; \lambda, \chi, \psi\right) \\
&= L_1\left({\mu}, \Sigma, {\gamma} \mid {x}_1, \cdots, {x}_n, w_1, \cdots, w_n\right)
+L_2\left(\lambda, \chi, \psi \mid w_1, \cdots, w_n\right)
\end{aligned}
$$

where \(X|W \sim N(\mu + w \gamma, w \Sigma)\).  

The conditional normal density is

$$
f_{{X}|W}(x|w) = \frac{e^{-\frac{{Q}(x)}{2w}}}{(2\pi w)^{d/2} |\Sigma|^{1/2}}
e^{(x - \mu)^{T} \Sigma^{-1} \gamma } 
e^{-\frac{w}{2} \gamma^{T} \Sigma^{-1} \gamma},
$$

with  

$$
Q(x) = (x - \mu)^{T} \Sigma^{-1} (x - \mu).
$$

---

### Separation of maximization

Thus, maximizing the likelihood can be separated:  

- \(L_1(\mu, \Sigma, \gamma)\) for location, scale, skewness parameters.  
- \(L_2(\lambda, \chi, \psi)\) for mixing parameters.  

#### \(L_1\)

$$
\begin{aligned}
L_1 &= \log(n) - \frac{n}{2} \log |\Sigma|
-\frac{d}{2} \sum_{i=1}^n \log w_i
+\sum_{i=1}^n (x_i-\mu)^{T} \Sigma^{-1} \gamma \\
&\quad -\frac{1}{2} \sum_{i=1}^n \frac{Q_i}{w_i}
-\frac{1}{2} \gamma^{T} \Sigma^{-1} \gamma \sum_{i=1}^n w_i
\end{aligned}
$$

#### \(L_2\)

$$
\begin{aligned}
L_2 &= (\lambda-1) \sum_{i=1}^n \log w_i
-\frac{\chi}{2} \sum_{i=1}^n w_i^{-1}
-\frac{\psi}{2} \sum_{i=1}^n w_i \\
&\quad -\frac{n \lambda}{2} \log \chi
+ \frac{n \lambda}{2} \log \psi
- n \log \left(2 K_\lambda(\sqrt{\chi \psi})\right)
\end{aligned}
$$

---

### Estimation from first-order conditions

From \(\partial L_1 / \partial \mu = 0\), \(\partial L_1 / \partial \gamma = 0\), \(\partial L_1 / \partial \Sigma = 0\):

$$
\gamma = \frac{n^{-1} \sum_{i=1}^n (x_i-\mu)}{n^{-1} \sum_{i=1}^n w_i},
$$

$$
\mu = \frac{n^{-1} \sum_{i=1}^n w_i^{-1} x_i - \gamma}{n^{-1} \sum_{i=1}^n w_i^{-1}},
$$

$$
\Sigma = \frac{1}{n} \sum_{i=1}^n w_i^{-1}(x_i-\mu)(x_i-\mu)^{T}
- \frac{1}{n} \sum_{i=1}^n w_i \gamma \gamma^{T}.
$$

Maximization of \(L_2\) yields the system:

$$
\frac{\partial L_2}{\partial \chi} = 0, \quad \frac{\partial L_2}{\partial \psi} = 0.
$$

This reduces to solving for \(\alpha = \sqrt{\chi \psi}\):

$$
n^2 K_{\lambda+1}(\alpha) K_{\lambda-1}(\alpha)
- \Big(\sum_{i=1}^n w_i^{-1}\Big)\Big(\sum_{i=1}^n w_i\Big) K_\lambda^2(\alpha) = 0.
$$

From which

$$
\psi = \frac{\alpha \sum_{i=1}^n w_i^{-1} K_{\lambda}(\alpha)}{n K_{\lambda - 1}(\alpha)},
\quad
\chi = \frac{\alpha^2}{\psi}.
$$

---

### Special cases

- **Normal Inverse Gaussian (NIG):** \(\lambda = -0.5\).  
- **Variance Gamma (VG):** \(\chi=0, \lambda > 0\).  
- **Skew-t:** \(\lambda = -\nu/2, \chi = \nu\).  

Each case yields simplifications in the root-solving equations.

---

### E-step

Introduce conditional expectations:

$$
\delta_i^{[k]}=E(W_i^{-1} \mid x_i; \xi^{[k]}), \quad
\eta_i^{[k]}=E(W_i \mid x_i; \xi^{[k]}), \quad
\zeta_i^{[k]}=E(\log W_i \mid x_i; \xi^{[k]}).
$$

These expectations are expressed in terms of modified Bessel functions \(K_\nu(\cdot)\).  

---

### M-step

Update rules at iteration \(k\):

$$
\gamma^{[k+1]} = \frac{\bar{x} - \mu^{[k]}}{\bar{\eta}^{[k]}},
$$

$$
\mu^{[k+1]} = \frac{n^{-1}\sum_{i=1}^n x_i \delta_i^{[k]} - \gamma^{[k+1]}}{\bar{\delta}^{[k]}},
$$

$$
\Sigma^{[k+1]} = \frac{1}{n} \sum_{i=1}^n \delta_i^{[k]} (x_i - \mu^{[k+1]})(x_i - \mu^{[k+1]})^T
- \bar{\eta}^{[k]} \gamma^{[k+1]} {\gamma^{[k+1]}}^T.
$$

To address the identification problem, normalize:

$$
\Sigma^{[k+1]} := \frac{c^{1/d} \Sigma^{[k+1]}}{|\Sigma^{[k+1]}|^{1/d}},
$$

where \(c = |\hat{\Sigma}_{\text{sample}}|\).

---

### Practical notes

- Iterative "online" updating of \(\mu, \gamma, \Sigma\) accelerates convergence.  
- This EM algorithm applies to GH, NIG, VG, and skew-t as special cases of the NMVM framework.  

