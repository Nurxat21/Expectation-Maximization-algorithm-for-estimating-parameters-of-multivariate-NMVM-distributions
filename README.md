# Expectation-Maximization Algorithm for Estimating Parameters of Multivariate Non-Normal Variable Mixture Distributions

Obtaining the optimal solution for the aforementioned objective function becomes increasingly challenging as the dimensionality of the data exceeds three. The core principle of the Expectation-Maximization (EM) algorithm is to enhance the following augmented log-likelihood function:


The generalized hyperbolic random variable can be represented as a conditional normal distribution, so the EM algorithm can be applied to such, in which most of the parameters $(\mathbf{\Sigma, \mu, \gamma})$ can be estimated similar with a Gaussian distribution when other parameters $(\lambda, \chi, \psi)$ are calibrated or assumed as some values. 

The EM algorithm framework of \cite{Mcneil_A_J_And_Frey_R_And_Embrechts_P_2015} for generalized hyperbolic distributions is the basis of our algorithm. Hu, in 2005, provided the unique algorithms for the limiting cases of generalized hyperbolic distributions parameters calibration \cite{Hu_Wenbo_2005}. In our part, we calibrate the estimation part of parameters $(\lambda, \chi, \psi)$ and use the online method in our estimation.  

If the data $\mathbf{X} = \mathbf{x_1}, \cdots, \mathbf{x_n}$ where $\mathbf{X} \in \mathcal{R}^{d \times n}$, we want to fit this random variable by multivariate generalized hyperbolic distributions, the parameters are denoted by $\mathbf{\xi} = (\lambda, \chi, \psi, \mathbf{\Sigma,\mu, \gamma})$, the log-likelihood function we want to maximize is 

$$
    \log L\left(\mathbf{\xi} ; \mathbf{x}_1, \cdots, \mathbf{x}_n\right)=\sum_{i=1}^n \log f_{\mathbf{x}_{\mathbf{i}}}\left(\mathbf{x}_i ; \mathbf{\xi}\right)
$$

It is hard to obtain the optimal solution of the above objective function directly if the data dimension is more than three. The main idea of the EM algorithm is to optimize the following augmented log-likelihood function:


$$
    \log \tilde{L}\left(\mathbf{\xi} ; \mathbf{x}_1, \cdots, \mathbf{x}_n, w_1, \cdots, w_n\right)=\sum_{i=1}^n \log f_{\mathbf{x}_{\mathbf{i}}, W_i}\left(\mathbf{x}_{\mathbf{i}}, w_i ; \mathbf{\xi}\right)
$$

where the latent mixing variables $\omega_1, \cdots, \omega_n$ are observable at the beginning. 


By the mean-variance mixture definition of generalized hyperbolic distributions, the log-likelihood function can be rewritten as

$$
\begin{aligned}
& \log \tilde{L}\left(\mathbf{\xi} ; \mathbf{x}_1, \cdots, \mathbf{x}_n, w_1, \cdots, w_n\right)= \\
& \quad \sum_{i=1}^n \log f_{\mathbf{x}_i \mid W_i}\left(\mathbf{x}_i \mid w_i ; \mathbf{\mu}, \Sigma, \mathbf{\gamma}\right)+ \\
& \quad \sum_{i=1}^n \log h_{W_i}\left(w_i ; \lambda, \chi, \psi\right)= \\
& \quad L_1\left(\mathbf{\mu}, \Sigma, \mathbf{\gamma} ; \mathbf{x}_1, \cdots, \mathbf{x}_n \mid w_1, \cdots, w_n\right)+L_2\left(\lambda, \chi, \psi ; w_1, \cdots, w_n\right)
\end{aligned}
$$

where $\mathbf{X}|W \sim N(\mathbf{\mu} + \omega \gamma, \omega \mathbf{\Sigma})$ and $f_{\mathbf{X}|W}(x|w)$ is the density of conditional normal distribution and $h(w)$ is the density function of $Z$ of (\ref{NMVM}). Following the same procedure in the proof of (\ref{Prop_GH}), the density of the conditional normal distribution can be shown as 
$$
    f_{\mathbf{X}|W}(x|w) = \frac{e^{\frac{-\mathcal{Q}(x)}{2\omega}}}{(2\pi \omega)^{\frac{d}{2}} |\mathbf{\Sigma}|^{\frac{1}{2}}} e^{(\mathbf{x - \mu})^{\top} \mathbf{\Sigma}^{-1} \mathbf{\gamma} } e^{-\frac{\omega}{2} \mathbf{\gamma^{\top} \Sigma^{-1} \gamma}}
$$

where 
$$
\mathcal{Q}(x) = \mathbf{(x - \mu)^{\top} \Sigma^{-1} (x - \mu)}
$$

From the (\ref{Log_likelihood_L1_L2}), the estimations of $(\mathbf{\mu, \Sigma, \gamma})$ and $(\lambda, \chi, \psi)$ can be separate by maximizing $L_1$ and $L_2$, respectively. 

$$
    \begin{aligned}
    & L_1\left(\mathbf{\mu}, \Sigma, \mathbf{\gamma} ; \mathbf{x}_1, \cdots, \mathbf{x}_n \mid w_1, \cdots, w_n\right)= \\
    & \quad-\frac{n}{2} \log |\Sigma|-\frac{d}{2} \sum_{i=1}^n \log w_i+\sum_{i=1}^n\left(\mathbf{x}_i-\mathbf{\mu}\right)^{\top} \Sigma^{-1} \mathbf{\gamma} \\
    & \quad-\frac{1}{2} \sum_{i=1}^n \frac{1}{w_i} \mathcal{Q}_i -\frac{1}{2} \mathbf{\gamma}^{\top} \Sigma^{-1} \mathbf{\gamma} \sum_{i=1}^n w_i
    \end{aligned}
$$

And the log-likelihood function $L_2$ can be written as:
$$
    \begin{aligned}
    & L_2\left(\lambda, \chi, \psi ; w_1, \cdots, w_n\right)= \\
    & \quad(\lambda-1) \sum_{i=1}^n \log w_i-\frac{\chi}{2} \sum_{i=1}^n w_i^{-1}-\frac{\psi}{2} \sum_{i=1}^n w_i-\frac{n \lambda}{2} \log \chi \\
    & \quad+\frac{n \lambda}{2} \log \psi-n \log \left(2 K_\lambda(\sqrt{\chi \psi})\right)
    \end{aligned}
$$

We take the partial derivative of $L_1$ concerning $\mathbf{\mu}$, $\mathbf{\gamma}$ and $\mathbf{\Sigma}$, which is the standard routine of optimization. From the partial derivatives equal to 0, we can get the following expressions:
$$
    \mathbf{\gamma}= \frac{n^{-1} \sum_{i=1}^n w_i^{-1}\left(\overline{\mathbf{x}}-\mathbf{x}_i\right)}{n^{-2}\left(\sum_{i=1}^n w_i\right)\left(\sum_{i=1}^n w_i^{-1}\right)-1} 
$$

$$
    \mathbf{\mu}= \frac{n^{-1} \sum_{i=1}^n w_i^{-1} \mathbf{x}_i-\mathbf{\gamma}}{n^{-1} \sum_{i=1}^n w_i^{-1}}
$$

$$
    \mathbf{\Sigma}=\frac{1}{n} \sum_{i=1}^n w_i^{-1}\left(\mathbf{x}_i-\mathbf{\mu}\right)\left(\mathbf{x}_i-\mathbf{\mu}\right)^{\top}-\frac{1}{n} \sum_{i=1}^n w_i \mathbf{\gamma} \mathbf{\gamma}^{\top}
$$

We maximize the (\ref{Log_L2}) to obtain the estimation of $\lambda, \chi \psi$. We first get the partial derivative concerning $\chi$ and $\psi$ and solve the next equations system. 

$$
\left\{ 
    \begin{aligned}
    & \frac{\partial L_2}{\partial \chi}=0 \\
    & \frac{\partial L_2}{\partial \psi}=0
    \end{aligned} 
\right.
$$
Solving (\ref{Equations_System_L2}) leads us to solve $\alpha = \sqrt{\chi \psi}$ from 
$$
    n^2 K_{\lambda + 1}(\alpha) K_{\lambda -1}(\alpha) - \sum_{i=1}^n w_i^{-1} \sum_{i=1}^n w_i K^2_{\lambda}(\alpha) = 0
$$

We can easily find the $\alpha$ by the root finding function of Python. Therefore, we can get the parameters as:
$$
    \begin{split}
        & \psi = \frac{\alpha \sum_{i=1}^n w_i^{-1} K_{\lambda}(\alpha)}{n K_{\lambda - 1}(\alpha)}\\
        & \chi = \frac{\alpha^2}{\psi}
    \end{split}
$$

Especially, when $\lambda=-0.5$, we have the normal inverse Gaussian distribution, and we are able to get $\alpha$ explicitly since $K_{-\lambda}(x)=K_\lambda(x)$ for any $\lambda$,
$$
    \alpha =\frac{2 \lambda}{1-n^{-2} \sum_{i=1}^n w_i \sum_{j=1}^n w_j^{-1}}
$$

When $\chi = 0$, and $\lambda >0$, in which it is VG distribution and we can get $\lambda$ by solving $\frac{\partial L_2}{\partial \lambda} = 0$ from the following equation:
$$
    \log (\lambda)-\log \left(n^{-1} \sum_{i=1}^n w_i\right)+n^{-1} \sum_{i=1}^n \log \left(w_i\right)-\phi(\lambda)=0.
$$

From the $\frac{\partial L_2}{\partial \psi} = 0$, and apply it into $\frac{\partial L_2}{\partial \lambda} = 0$, thus, we get (\ref{Solving_lambda_VG}), and the $\psi = \frac{2\lambda}{n^{-1} \sum_{i=1}^n w_i}$. 

If we continue to assume $\lambda = -\frac{\nu}{2}$ and $\chi = \nu$, we obtain skew-t distribution with degree of freedom $\nu$. We still follows the standard routine, in which the only parameter $\nu$ can be solved by 
$$
    log(\frac{\nu}{2}) + 1 - n^{-1} \sum_{i=1}^n w_i^{-1}-n^{-1} \sum_{i=1}^n \log \left(w_i\right) - \phi(\frac{\nu}{2}) = 0,
$$
As we mentioned, the latent mixing variables $\omega_i$ are not observable. Therefore, an iteration procedure consisting of E-step and M-step are needed, in which the E-step is called the estimation step. In E-step, the conditional expectation of the augmented log-likelihood function given current parameter estimates and sample data is calculated. Suppose that we are at step $k$, we need to calculate the following conditional expectation and get a new objective function to be maximized.

$$
    Q\left(\mathbf{\xi} ; \mathbf{\xi}^{[k]}\right)=E\left(\log \tilde{L}\left(\mathbf{\xi} ; \mathbf{x}_1, \cdots, \mathbf{x}_n, W_1, \cdots, W_n\right) \mid \mathbf{x}_1, \cdots, \mathbf{x}_n ; \mathbf{\xi}^{[k]}\right)
$$
In M-step, we maximize the $Q$ function above to get updated estimates $\xi^{[k+1]}$. From the (\ref{Log_L1}) and (\ref{Log_L2}), it shows that updating the $\omega_i$, $\omega^{-1}_i$ and $log(\omega_i)$ is equivalent to the conditional estimates $E(\mathbf{W_i|x_i;\xi^{[k]}})$, $E(\mathbf{W^{-1}_i|x_i;\xi^{[k]}})$ and $E(log\mathbf{(W_i)|x_i;\xi^{[k]}})$. Those conditional expectations can be calculated by following conditional density function:
$$
    f_{W \mid \mathbf{X}}(w \mid \mathbf{x} ; \mathbf{\xi})=\frac{f(\mathbf{x} \mid w ; \mathbf{\xi}) h(w ; \mathbf{\xi})}{f(\mathbf{x} ; \mathbf{\xi})}
$$
where we can get 
$$
    W_i \mid \mathbf{X_i} \sim N^- (\lambda - \frac{d}{2}, \mathcal{Q}(x_i) + \chi, \psi + \mathbf{\gamma^{\top} \Sigma^{-1} \gamma})
$$

For convenience, we use the standard notation of \cite{protassov2004based, Mcneil_A_J_And_Frey_R_And_Embrechts_P_2015, Hu_Wenbo_2005}, which shows as :
$$
    \delta_i^{[\cdot]}=E\left(W_i^{-1} \mid \mathbf{x}_i ; \mathbf{\xi}^{[\cdot]}\right), \eta_i^{[\cdot]}=E\left(W_i \mid \mathbf{x}_i ; \mathbf{\xi}^{[\cdot]}\right), \xi_i^{[\cdot]}=E\left(\log \left(W_i\right) \mid \mathbf{x}_i ; \mathbf{\xi}^{[\cdot]}\right),
$$
and 

$$
    \bar{\delta}=\frac{1}{n} \sum_1^n \delta_i, \bar{\eta}=\frac{1}{n} \sum_1^n \eta_i, \bar{\xi}=\frac{1}{n} \sum_1^n \zeta_i
$$

For the generalized hyperbolic distributions, we have 

$$
\delta_i^{[k]}=\left(\frac{\mathcal{Q}^{[k]}_i+\chi^{[k]}}{\psi^{[k]}+\mathbf{\gamma}^{[k]} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}}\right)^{-\frac{1}{2}} \frac{K_{\lambda-\frac{d}{2}-1}\left(\sqrt{\left(\mathcal{Q}^{[k]}_i+\chi^{[k]}\right)\left(\psi^{[k]}+\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]-1} \mathbf{\gamma}^{[k]}\right)}\right)}{K_{\lambda-\frac{d}{2}}\left(\sqrt{\left(\mathcal{Q}^{[k]}_i+\chi^{[k]}\right)\left(\psi^{[k]}+\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}\right)}\right)}\\
$$

$$
    \eta_i^{[k]}=\left(\frac{\mathcal{Q}^{[k]}_i+\chi^{[k]}}{\psi^{[k]}+\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}}\right)^{\frac{1}{2}} \frac{K_{\lambda-\frac{d}{2}+1}\left(\sqrt{\left(\mathcal{Q}^{[k]}_i+\chi^{[k]}\right)\left(\psi^{[k]}+\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}\right)}\right)}{K_{\lambda-\frac{d}{2}}\left(\sqrt{\left(\mathcal{Q}^{[k]}_i+\chi^{[k]}\right)\left(\psi^{[k]}+\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]-1} \mathbf{\gamma}^{[k]}\right)}\right)}
$$

$$
\begin{aligned}
\zeta_i^{[k]} & =\frac{1}{2} \log \left(\frac{\mathcal{Q}^{[k]}_i+\chi^{[k]}}{\psi^{[k]}+\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}}\right)+ \\
& \frac{\left.\frac{\partial K_{\lambda-\frac{d}{2}+\alpha}\left(\sqrt{\left(\mathcal{Q}^{[k]}_i+\chi^{[k]}\right)\left(\psi^{[k]}+\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}\right)}\right)}{\partial \alpha}\right|_{\alpha=0}}{K_{\lambda-\frac{d}{2}}\left(\sqrt{\left(\mathcal{Q}^{[k]}_i+\chi^{[k]}\right)\left(\psi^{[k]}+\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}\right)}\right)} .
\end{aligned}
$$

For the variance gamma distribution, we just need to set $\chi = 0$ in above equations, in which we have 

$$
    W_i \mid \mathbf{X_i} \sim N^- (\lambda - \frac{d}{2}, \mathcal{Q}(x_i), \psi + \mathbf{\gamma^{\top} \Sigma^{-1} \gamma})
$$

For the multivariate Skew-t distribution, we have 

$$
    W_i \mid \mathbf{X_i} \sim N^- (-\frac{d + \nu}{2}, \mathcal{Q}(x_i) + \nu, \mathbf{\gamma^{\top} \Sigma^{-1} \gamma})
$$

where $\delta^{[k]}$, $\eta^{[k]}$ and $\zeta_i^{[k]}$ show as 
$$
    \delta_i^{[k]}=\left(\frac{\mathcal{Q}^{[k]}_i+\nu^{[k]}}{\mathbf{\gamma}^{[k]^2} \Sigma^{[k]-1} \mathbf{\gamma}^{[k]}}\right)^{-\frac{1}{2}} \frac{K_{\frac{\nu+d+2}{2}}\left(\sqrt{\left(\mathcal{Q}^{[k]}_i+\nu^{[k]}\right)\left(\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}\right)}\right)}{K_{\frac{\nu+d}{2}}\left(\sqrt{\left(\mathcal{Q}^{[k]}_i+\nu^{[k]}\right)\left(\mathbf{\gamma}^{[k]^{\top} \Sigma^{[k]]^{-1}} \mathbf{\gamma}^{[k]}}\right)}\right)}
$$

$$
    \eta_i^{[k]}=\left(\frac{\mathcal{Q}^{[k]}+\nu^{[k]}}{\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}}\right)^{\frac{1}{2}} \frac{K_{\frac{\nu+d-2}{2}}\left(\sqrt{\left(\mathcal{Q}^{[k]}+\nu^{[k]}\right)\left(\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}\right)}\right)}{K_{\frac{\nu+d}{2}}\left(\sqrt{\left(\mathcal{Q}^{[k]}+\nu^{[k]}\right)\left(\mathbf{\gamma}^{\left.[k]^{\top} \Sigma^{[k]}\right]^{-1}} \mathbf{\gamma}^{[k]}\right)}\right)}
$$

$$
\begin{aligned}
\zeta_i^{[k]} & =\frac{1}{2} \log \left(\frac{\mathcal{Q}^{[k]}+\nu^{[k]}}{\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}}\right)+ \\
& \frac{\left.\frac{\partial K_{-\frac{\nu+d}{2}+\alpha}\left(\sqrt{\left(\mathcal{Q}^{[k]}+\nu^{[k]}\right)\left(\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]^{-1}} \mathbf{\gamma}^{[k]}\right)}\right)}{\partial \alpha}\right|_{\alpha=0}}{K_{\frac{v_{+d}}{2}}\left(\sqrt{\left(\mathcal{Q}^{[k]}+\nu^{[k]}\right)\left(\mathbf{\gamma}^{[k]^{\top}} \Sigma^{[k]-1} \mathbf{\gamma}^{[k]}\right)}\right)} .
\end{aligned}
$$

In the M-step, we need to replace the latent variables $\omega^{-1}_i$ by $\delta^{[k]}_i$, $\omega_i$ by $\eta^{[k]}_i$, $log(\omega_i)$ by $\zeta^{[k]}_i$ in the maximization. Thus, maximizing the conditional expectation of $L_1$, we can get the k-step estimations of $\mathbf{\gamma}^{[k+1]}$, $\mathbf{\mu}^{[k+1]}$ and $\mathbf{\Sigma}^{[k+1]}$. During the maximizing of conditional expectation of $L_2$, we need to solve (\ref{Solving_Equation_alpha}) to obtain the $\alpha^{[k+1]}$. When we get the $\alpha^{[k+1]}$, we can update the values of $\chi^{[k+1]}$ and $\psi^{[k+1]}$. 


When we calibrating the parameter of GH distribution, we faced to an identification problem. In \cite{Mcneil_A_J_And_Frey_R_And_Embrechts_P_2015}, they used a fixed number $c$ to be the determinant of $\Sigma$, in which the number $c$ is the determinate of sample covariance matrix to solve such problem by using (\ref{Sigma_GH_estimation}) and set
$$
    \Sigma^{[k+1]}:=\frac{c^{1 / d} \Sigma^{[k+1]}}{\left|\Sigma^{[k+1]}\right|^{1 / d}}
$$

As mentioned in \cite{Hu_Wenbo_2005}, when $|\lambda|$ is large, the (\ref{Solving_Equation_alpha}) may be not equalled to zero so that we will minimize the square root of (\ref{Solving_Equation_alpha}) to update the $\alpha^{[k+1]}$. In \cite{Hu_Wenbo_2005}, he set $\chi$ or $\psi$ to be constant, when $|\lambda|$ is large. However, different choices of constant $\chi$ or $\psi$ would lead to different estimating speed, and it might crash in some special constant values. 

The EM algorithm iteratively updates the parameters to maximize the likelihood function. The iterative process begins with initial estimates for the parameters \(\xi = (\lambda, \chi, \psi, \Sigma, \mu, \gamma)\). In the Expectation Step (E-step), the expected value of the complete-data log-likelihood function is calculated with respect to the current parameter estimates, involving the computation of the expected values of the latent mixing variables \(w_i\) given the observed data. The Maximization Step (M-step) follows, where the expected complete-data log-likelihood function obtained in the E-step is maximized to update the parameter estimates, including the location parameter \(\mu\), the dispersion parameter \(\Sigma\) using the optimization (\ref{Sigma_K_1})to ensure positive definiteness, the skewness parameter \(\gamma\), and the parameters of the generalized inverse Gaussian (GIG) distribution \(\lambda\), \(\chi\), and \(\psi\). 
