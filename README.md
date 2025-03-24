# Expectation-Maximization Algorithm for Estimating Parameters of Multivariate Non-Normal Variable Mixture Distributions

Obtaining the optimal solution for the aforementioned objective function becomes increasingly challenging as the dimensionality of the data exceeds three. The core principle of the Expectation-Maximization (EM) algorithm is to enhance the following augmented log-likelihood function:

Many researchers studied the univariate generalized hyperbolic
distribution in modelling financial data. However, the study on the
multivariate generalized hyperbolic distributions was incomplete in
financial modelling fields. Maximum log-likelihood estimation is the
most famous method for estimating the parameters of distributions. In ,
they developed a program HYP to fit the hyperbolic distribution by
maximum log-likelihood estimation, and they can only calibrate the
hyperbolic distribution in less than three dimensions. Moreover, Prause
provided detailed derivations of the derivatives of the log-likelihood
function for generalized hyperbolic distributions in his dissertation,
and he applied HY P to calibrate the three-dimensional generalized
hyperbolic distributions .

The generalized hyperbolic random variable can be represented as a
conditional normal distribution, so the EM algorithm can be applied to
such, in which most of the parameters (**Σ****,** **μ****,** **γ**) can
be estimated similar with a Gaussian distribution when other parameters
(*λ*,*χ*,*ψ*) are calibrated or assumed as some values.

The EM algorithm framework of for generalized hyperbolic distributions
is the basis of our algorithm. Hu, in 2005, provided the unique
algorithms for the limiting cases of generalized hyperbolic
distributions parameters calibration . In our part, we calibrate the
estimation part of parameters (*λ*,*χ*,*ψ*) and use the online method in
our estimation.

If the data **X** = **x**<sub>**1**</sub>, ⋯, **x**<sub>**n**</sub>
where **X** ∈ ℛ<sup>*d* × *n*</sup>, we want to fit this random variable
by multivariate generalized hyperbolic distributions, the parameters are
denoted by **ξ** = (*λ*,*χ*,*ψ*,**Σ****,** **μ****,** **γ**), the
log-likelihood function we want to maximize is
$$\\label{log_likelihood_function}
    \\log L\\left(\\mathbf{\\xi} ; \\mathbf{x}\_1, \\cdots, \\mathbf{x}\_n\\right)=\\sum\_{i=1}^n \\log f\_{\\mathbf{x}\_{\\mathbf{i}}}\\left(\\mathbf{x}\_i ; \\mathbf{\\xi}\\right)$$

It is hard to obtain the optimal solution of the above objective
function directly if the data dimension is more than three. The main
idea of the EM algorithm is to optimize the following augmented
log-likelihood function:

$$\\label{augmented_log_likelihood_function}
    \\log \\tilde{L}\\left(\\mathbf{\\xi} ; \\mathbf{x}\_1, \\cdots, \\mathbf{x}\_n, w_1, \\cdots, w_n\\right)=\\sum\_{i=1}^n \\log f\_{\\mathbf{x}\_{\\mathbf{i}}, W_i}\\left(\\mathbf{x}\_{\\mathbf{i}}, w_i ; \\mathbf{\\xi}\\right)$$
where the latent mixing variables *ω*<sub>1</sub>, ⋯, *ω*<sub>*n*</sub>
are observable at the beginning.

By the mean-variance mixture definition of generalized hyperbolic
distributions, the log-likelihood function can be rewritten as

$$\\label{Log_likelihood_L1_L2}
\\begin{aligned}
& \\log \\tilde{L}\\left(\\mathbf{\\xi} ; \\mathbf{x}\_1, \\cdots, \\mathbf{x}\_n, w_1, \\cdots, w_n\\right)= \\\\
& \\quad \\sum\_{i=1}^n \\log f\_{\\mathbf{x}\_i \\mid W_i}\\left(\\mathbf{x}\_i \\mid w_i ; \\mathbf{\\mu}, \\Sigma, \\mathbf{\\gamma}\\right)+ \\\\
& \\quad \\sum\_{i=1}^n \\log h\_{W_i}\\left(w_i ; \\lambda, \\chi, \\psi\\right)= \\\\
& \\quad L_1\\left(\\mathbf{\\mu}, \\Sigma, \\mathbf{\\gamma} ; \\mathbf{x}\_1, \\cdots, \\mathbf{x}\_n \\mid w_1, \\cdots, w_n\\right)+L_2\\left(\\lambda, \\chi, \\psi ; w_1, \\cdots, w_n\\right)
\\end{aligned}$$
where **X**\|*W* ∼ *N*(**μ**+*ω**γ*,*ω***Σ**) and
*f*<sub>**X**\|*W*</sub>(*x*\|*w*) is the density of conditional normal
distribution and *h*(*w*) is the density function of *Z* of
(<a href="#NMVM" data-reference-type="ref"
data-reference="NMVM">[NMVM]</a>). Following the same procedure in the
proof of (<a href="#Prop_GH" data-reference-type="ref"
data-reference="Prop_GH">[Prop_GH]</a>), the density of the conditional
normal distribution can be shown as
$$\\label{PDF_Conditional_Normal}
    f\_{\\mathbf{X}\|W}(x\|w) = \\frac{e^{\\frac{-\\mathcal{Q}(x)}{2\\omega}}}{(2\\pi \\omega)^{\\frac{d}{2}} \|\\mathbf{\\Sigma}\|^{\\frac{1}{2}}} e^{(\\mathbf{x - \\mu})^{\\top} \\mathbf{\\Sigma}^{-1} \\mathbf{\\gamma} } e^{-\\frac{\\omega}{2} \\mathbf{\\gamma^{\\top} \\Sigma^{-1} \\gamma}}$$

where
𝒬(*x*) = **(****x****−****μ****)**<sup>**⊤**</sup>**Σ**<sup>**−****1**</sup>**(****x****−****μ****)**

From the (<a href="#Log_likelihood_L1_L2" data-reference-type="ref"
data-reference="Log_likelihood_L1_L2">[Log_likelihood_L1_L2]</a>), the
estimations of (**μ****,** **Σ****,** **γ**) and (*λ*,*χ*,*ψ*) can be
separate by maximizing *L*<sub>1</sub> and *L*<sub>2</sub>,
respectively.

$$\\label{Log_L1}
    \\begin{aligned}
    & L_1\\left(\\mathbf{\\mu}, \\Sigma, \\mathbf{\\gamma} ; \\mathbf{x}\_1, \\cdots, \\mathbf{x}\_n \\mid w_1, \\cdots, w_n\\right)= \\\\
    & \\quad-\\frac{n}{2} \\log \|\\Sigma\|-\\frac{d}{2} \\sum\_{i=1}^n \\log w_i+\\sum\_{i=1}^n\\left(\\mathbf{x}\_i-\\mathbf{\\mu}\\right)^{\\top} \\Sigma^{-1} \\mathbf{\\gamma} \\\\
    & \\quad-\\frac{1}{2} \\sum\_{i=1}^n \\frac{1}{w_i} \\mathcal{Q}\_i -\\frac{1}{2} \\mathbf{\\gamma}^{\\top} \\Sigma^{-1} \\mathbf{\\gamma} \\sum\_{i=1}^n w_i
    \\end{aligned}$$

And the log-likelihood function *L*<sub>2</sub> can be written as:
$$\\label{Log_L2}
    \\begin{aligned}
    & L_2\\left(\\lambda, \\chi, \\psi ; w_1, \\cdots, w_n\\right)= \\\\
    & \\quad(\\lambda-1) \\sum\_{i=1}^n \\log w_i-\\frac{\\chi}{2} \\sum\_{i=1}^n w_i^{-1}-\\frac{\\psi}{2} \\sum\_{i=1}^n w_i-\\frac{n \\lambda}{2} \\log \\chi \\\\
    & \\quad+\\frac{n \\lambda}{2} \\log \\psi-n \\log \\left(2 K\_\\lambda(\\sqrt{\\chi \\psi})\\right)
    \\end{aligned}$$

We take the partial derivative of *L*<sub>1</sub> concerning **μ**,
**γ** and **Σ**, which is the standard routine of optimization. From the
partial derivatives equal to 0, we can get the following expressions:
$$\\label{gamma_GH_estimation}
    \\mathbf{\\gamma}= \\frac{n^{-1} \\sum\_{i=1}^n w_i^{-1}\\left(\\overline{\\mathbf{x}}-\\mathbf{x}\_i\\right)}{n^{-2}\\left(\\sum\_{i=1}^n w_i\\right)\\left(\\sum\_{i=1}^n w_i^{-1}\\right)-1}$$

$$\\label{mu_GH_estimation}
    \\mathbf{\\mu}= \\frac{n^{-1} \\sum\_{i=1}^n w_i^{-1} \\mathbf{x}\_i-\\mathbf{\\gamma}}{n^{-1} \\sum\_{i=1}^n w_i^{-1}}$$

$$\\label{Sigma_GH_estimation}
    \\mathbf{\\Sigma}=\\frac{1}{n} \\sum\_{i=1}^n w_i^{-1}\\left(\\mathbf{x}\_i-\\mathbf{\\mu}\\right)\\left(\\mathbf{x}\_i-\\mathbf{\\mu}\\right)^{\\top}-\\frac{1}{n} \\sum\_{i=1}^n w_i \\mathbf{\\gamma} \\mathbf{\\gamma}^{\\top}$$

We maximize the (<a href="#Log_L2" data-reference-type="ref"
data-reference="Log_L2">[Log_L2]</a>) to obtain the estimation of
*λ*, *χ**ψ*. We first get the partial derivative concerning *χ* and *ψ*
and solve the next equations system.

$$\\label{Equations_System_L2}
\\left\\{ 
    \\begin{aligned}
    & \\frac{\\partial L_2}{\\partial \\chi}=0 \\\\
    & \\frac{\\partial L_2}{\\partial \\psi}=0
    \\end{aligned} 
\\right.$$
Solving (<a href="#Equations_System_L2" data-reference-type="ref"
data-reference="Equations_System_L2">[Equations_System_L2]</a>) leads us
to solve $\\alpha = \\sqrt{\\chi \\psi}$ from
$$\\label{Solving_Equation_alpha}
    n^2 K\_{\\lambda + 1}(\\alpha) K\_{\\lambda -1}(\\alpha) - \\sum\_{i=1}^n w_i^{-1} \\sum\_{i=1}^n w_i K^2\_{\\lambda}(\\alpha) = 0$$

We can easily find the *α* by the root finding function of Python.
Therefore, we can get the parameters as:
$$\\begin{split}
        & \\psi = \\frac{\\alpha \\sum\_{i=1}^n w_i^{-1} K\_{\\lambda}(\\alpha)}{n K\_{\\lambda - 1}(\\alpha)}\\\\
        & \\chi = \\frac{\\alpha^2}{\\psi}
    \\end{split}$$

Especially, when *λ* =  − 0.5, we have the normal inverse Gaussian
distribution, and we are able to get *α* explicitly since
*K*<sub>−*λ*</sub>(*x*) = *K*<sub>*λ*</sub>(*x*) for any *λ*,
$$\\alpha =\\frac{2 \\lambda}{1-n^{-2} \\sum\_{i=1}^n w_i \\sum\_{j=1}^n w_j^{-1}}$$

When *χ* = 0, and *λ* \> 0, in which it is VG distribution and we can
get *λ* by solving $\\frac{\\partial L_2}{\\partial \\lambda} = 0$ from
the following equation:
$$\\label{Solving_lambda_VG}
    \\log (\\lambda)-\\log \\left(n^{-1} \\sum\_{i=1}^n w_i\\right)+n^{-1} \\sum\_{i=1}^n \\log \\left(w_i\\right)-\\phi(\\lambda)=0.$$

From the $\\frac{\\partial L_2}{\\partial \\psi} = 0$, and apply it into
$\\frac{\\partial L_2}{\\partial \\lambda} = 0$, thus, we get
(<a href="#Solving_lambda_VG" data-reference-type="ref"
data-reference="Solving_lambda_VG">[Solving_lambda_VG]</a>), and the
$\\psi = \\frac{2\\lambda}{n^{-1} \\sum\_{i=1}^n w_i}$.

If we continue to assume $\\lambda = -\\frac{\\nu}{2}$ and *χ* = *ν*, we
obtain skew-t distribution with degree of freedom *ν*. We still follows
the standard routine, in which the only parameter *ν* can be solved by
$$\\label{Solving_v\_Skew-t}
    log(\\frac{\\nu}{2}) + 1 - n^{-1} \\sum\_{i=1}^n w_i^{-1}-n^{-1} \\sum\_{i=1}^n \\log \\left(w_i\\right) - \\phi(\\frac{\\nu}{2}) = 0,$$
As we mentioned, the latent mixing variables *ω*<sub>*i*</sub> are not
observable. Therefore, an iteration procedure consisting of E-step and
M-step are needed, in which the E-step is called the estimation step. In
E-step, the conditional expectation of the augmented log-likelihood
function given current parameter estimates and sample data is
calculated. Suppose that we are at step *k*, we need to calculate the
following conditional expectation and get a new objective function to be
maximized.

*Q*(**ξ**;**ξ**<sup>\[*k*\]</sup>) = *E*(log*L̃*(**ξ**;**x**<sub>1</sub>,⋯,**x**<sub>*n*</sub>,*W*<sub>1</sub>,⋯,*W*<sub>*n*</sub>)∣**x**<sub>1</sub>,⋯,**x**<sub>*n*</sub>;**ξ**<sup>\[*k*\]</sup>)
In M-step, we maximize the *Q* function above to get updated estimates
*ξ*<sup>\[*k*+1\]</sup>. From the
(<a href="#Log_L1" data-reference-type="ref"
data-reference="Log_L1">[Log_L1]</a>) and
(<a href="#Log_L2" data-reference-type="ref"
data-reference="Log_L2">[Log_L2]</a>), it shows that updating the
*ω*<sub>*i*</sub>, *ω*<sub>*i*</sub><sup>−1</sup> and
*l**o**g*(*ω*<sub>*i*</sub>) is equivalent to the conditional estimates
*E*(**W**<sub>**i**</sub>**\|****x**<sub>**i**</sub>**;** **ξ**<sup>**\[****k****\]**</sup>),
*E*(**W**<sub>**i**</sub><sup>**−****1**</sup>**\|****x**<sub>**i**</sub>**;** **ξ**<sup>**\[****k****\]**</sup>)
and
*E*(*l**o**g***(****W**<sub>**i**</sub>**)****\|****x**<sub>**i**</sub>**;** **ξ**<sup>**\[****k****\]**</sup>).
Those conditional expectations can be calculated by following
conditional density function:
$$f\_{W \\mid \\mathbf{X}}(w \\mid \\mathbf{x} ; \\mathbf{\\xi})=\\frac{f(\\mathbf{x} \\mid w ; \\mathbf{\\xi}) h(w ; \\mathbf{\\xi})}{f(\\mathbf{x} ; \\mathbf{\\xi})}$$
where we can get
$$W_i \\mid \\mathbf{X_i} \\sim N^- (\\lambda - \\frac{d}{2}, \\mathcal{Q}(x_i) + \\chi, \\psi + \\mathbf{\\gamma^{\\top} \\Sigma^{-1} \\gamma})$$

For convenience, we use the standard notation of , which shows as :
*δ*<sub>*i*</sub><sup>\[⋅\]</sup> = *E*(*W*<sub>*i*</sub><sup>−1</sup>∣**x**<sub>*i*</sub>;**ξ**<sup>\[⋅\]</sup>), *η*<sub>*i*</sub><sup>\[⋅\]</sup> = *E*(*W*<sub>*i*</sub>∣**x**<sub>*i*</sub>;**ξ**<sup>\[⋅\]</sup>), *ξ*<sub>*i*</sub><sup>\[⋅\]</sup> = *E*(log(*W*<sub>*i*</sub>)∣**x**<sub>*i*</sub>;**ξ**<sup>\[⋅\]</sup>),
and
$$\\bar{\\delta}=\\frac{1}{n} \\sum_1^n \\delta_i, \\bar{\\eta}=\\frac{1}{n} \\sum_1^n \\eta_i, \\bar{\\xi}=\\frac{1}{n} \\sum_1^n \\zeta_i$$
For the generalized hyperbolic distributions, we have
$$\\delta_i^{\[k\]}=\\left(\\frac{\\mathcal{Q}^{\[k\]}\_i+\\chi^{\[k\]}}{\\psi^{\[k\]}+\\mathbf{\\gamma}^{\[k\]} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}}\\right)^{-\\frac{1}{2}} \\frac{K\_{\\lambda-\\frac{d}{2}-1}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}\_i+\\chi^{\[k\]}\\right)\\left(\\psi^{\[k\]}+\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]-1} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)}{K\_{\\lambda-\\frac{d}{2}}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}\_i+\\chi^{\[k\]}\\right)\\left(\\psi^{\[k\]}+\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)}\\\\$$

$$\\eta_i^{\[k\]}=\\left(\\frac{\\mathcal{Q}^{\[k\]}\_i+\\chi^{\[k\]}}{\\psi^{\[k\]}+\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}}\\right)^{\\frac{1}{2}} \\frac{K\_{\\lambda-\\frac{d}{2}+1}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}\_i+\\chi^{\[k\]}\\right)\\left(\\psi^{\[k\]}+\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)}{K\_{\\lambda-\\frac{d}{2}}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}\_i+\\chi^{\[k\]}\\right)\\left(\\psi^{\[k\]}+\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]-1} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)}$$

$$\\begin{aligned}
\\zeta_i^{\[k\]} & =\\frac{1}{2} \\log \\left(\\frac{\\mathcal{Q}^{\[k\]}\_i+\\chi^{\[k\]}}{\\psi^{\[k\]}+\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}}\\right)+ \\\\
& \\frac{\\left.\\frac{\\partial K\_{\\lambda-\\frac{d}{2}+\\alpha}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}\_i+\\chi^{\[k\]}\\right)\\left(\\psi^{\[k\]}+\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)}{\\partial \\alpha}\\right\|\_{\\alpha=0}}{K\_{\\lambda-\\frac{d}{2}}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}\_i+\\chi^{\[k\]}\\right)\\left(\\psi^{\[k\]}+\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)} .
\\end{aligned}$$

For the variance gamma distribution, we just need to set *χ* = 0 in
above equations, in which we have
$$W_i \\mid \\mathbf{X_i} \\sim N^- (\\lambda - \\frac{d}{2}, \\mathcal{Q}(x_i), \\psi + \\mathbf{\\gamma^{\\top} \\Sigma^{-1} \\gamma})$$
For the multivariate Skew-t distribution, we have
$$W_i \\mid \\mathbf{X_i} \\sim N^- (-\\frac{d + \\nu}{2}, \\mathcal{Q}(x_i) + \\nu, \\mathbf{\\gamma^{\\top} \\Sigma^{-1} \\gamma})$$
where *δ*<sup>\[*k*\]</sup>, *η*<sup>\[*k*\]</sup> and
*ζ*<sub>*i*</sub><sup>\[*k*\]</sup> show as
$$\\delta_i^{\[k\]}=\\left(\\frac{\\mathcal{Q}^{\[k\]}\_i+\\nu^{\[k\]}}{\\mathbf{\\gamma}^{\[k\]^2} \\Sigma^{\[k\]-1} \\mathbf{\\gamma}^{\[k\]}}\\right)^{-\\frac{1}{2}} \\frac{K\_{\\frac{\\nu+d+2}{2}}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}\_i+\\nu^{\[k\]}\\right)\\left(\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)}{K\_{\\frac{\\nu+d}{2}}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}\_i+\\nu^{\[k\]}\\right)\\left(\\mathbf{\\gamma}^{\[k\]^{\\top} \\Sigma^{\[k\]\]^{-1}} \\mathbf{\\gamma}^{\[k\]}}\\right)}\\right)}$$

$$\\eta_i^{\[k\]}=\\left(\\frac{\\mathcal{Q}^{\[k\]}+\\nu^{\[k\]}}{\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}}\\right)^{\\frac{1}{2}} \\frac{K\_{\\frac{\\nu+d-2}{2}}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}+\\nu^{\[k\]}\\right)\\left(\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)}{K\_{\\frac{\\nu+d}{2}}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}+\\nu^{\[k\]}\\right)\\left(\\mathbf{\\gamma}^{\\left.\[k\]^{\\top} \\Sigma^{\[k\]}\\right\]^{-1}} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)}$$

$$\\begin{aligned}
\\zeta_i^{\[k\]} & =\\frac{1}{2} \\log \\left(\\frac{\\mathcal{Q}^{\[k\]}+\\nu^{\[k\]}}{\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}}\\right)+ \\\\
& \\frac{\\left.\\frac{\\partial K\_{-\\frac{\\nu+d}{2}+\\alpha}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}+\\nu^{\[k\]}\\right)\\left(\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]^{-1}} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)}{\\partial \\alpha}\\right\|\_{\\alpha=0}}{K\_{\\frac{v\_{+d}}{2}}\\left(\\sqrt{\\left(\\mathcal{Q}^{\[k\]}+\\nu^{\[k\]}\\right)\\left(\\mathbf{\\gamma}^{\[k\]^{\\top}} \\Sigma^{\[k\]-1} \\mathbf{\\gamma}^{\[k\]}\\right)}\\right)} .
\\end{aligned}$$

In the M-step, we need to replace the latent variables
*ω*<sub>*i*</sub><sup>−1</sup> by *δ*<sub>*i*</sub><sup>\[*k*\]</sup>,
*ω*<sub>*i*</sub> by *η*<sub>*i*</sub><sup>\[*k*\]</sup>,
*l**o**g*(*ω*<sub>*i*</sub>) by *ζ*<sub>*i*</sub><sup>\[*k*\]</sup> in
the maximization. Thus, maximizing the conditional expectation of
*L*<sub>1</sub>, we can get the k-step estimations of
**γ**<sup>\[*k*+1\]</sup>, **μ**<sup>\[*k*+1\]</sup> and
**Σ**<sup>\[*k*+1\]</sup>. During the maximizing of conditional
expectation of *L*<sub>2</sub>, we need to solve
(<a href="#Solving_Equation_alpha" data-reference-type="ref"
data-reference="Solving_Equation_alpha">[Solving_Equation_alpha]</a>) to
obtain the *α*<sup>\[*k*+1\]</sup>. When we get the
*α*<sup>\[*k*+1\]</sup>, we can update the values of
*χ*<sup>\[*k*+1\]</sup> and *ψ*<sup>\[*k*+1\]</sup>.

When we calibrating the parameter of GH distribution, we faced to an
identification problem. In , they used a fixed number *c* to be the
determinant of *Σ*, in which the number *c* is the determinate of sample
covariance matrix to solve such problem by using
(<a href="#Sigma_GH_estimation" data-reference-type="ref"
data-reference="Sigma_GH_estimation">[Sigma_GH_estimation]</a>) and set
$$\\label{Sigma_K\_1}
    \\Sigma^{\[k+1\]}:=\\frac{c^{1 / d} \\Sigma^{\[k+1\]}}{\\left\|\\Sigma^{\[k+1\]}\\right\|^{1 / d}}$$

As mentioned in , when \|*λ*\| is large, the
(<a href="#Solving_Equation_alpha" data-reference-type="ref"
data-reference="Solving_Equation_alpha">[Solving_Equation_alpha]</a>)
may be not equalled to zero so that we will minimize the square root of
(<a href="#Solving_Equation_alpha" data-reference-type="ref"
data-reference="Solving_Equation_alpha">[Solving_Equation_alpha]</a>) to
update the *α*<sup>\[*k*+1\]</sup>. In , he set *χ* or *ψ* to be
constant, when \|*λ*\| is large. However, different choices of constant
*χ* or *ψ* would lead to different estimating speed, and it might crash
in some special constant values.

The EM algorithm iteratively updates the parameters to maximize the
likelihood function. The iterative process begins with initial estimates
for the parameters *ξ* = (*λ*,*χ*,*ψ*,*Σ*,*μ*,*γ*). In the Expectation
Step (E-step), the expected value of the complete-data log-likelihood
function is calculated with respect to the current parameter estimates,
involving the computation of the expected values of the latent mixing
variables *w*<sub>*i*</sub> given the observed data. The Maximization
Step (M-step) follows, where the expected complete-data log-likelihood
function obtained in the E-step is maximized to update the parameter
estimates, including the location parameter *μ*, the dispersion
parameter *Σ* using the optimization
(<a href="#Sigma_K_1" data-reference-type="ref"
data-reference="Sigma_K_1">[Sigma_K_1]</a>)to ensure positive
definiteness, the skewness parameter *γ*, and the parameters of the
generalized inverse Gaussian (GIG) distribution *λ*, *χ*, and *ψ*.

After updating, the parameters
(*λ*<sup>\[*k*+1\]</sup>,*χ*<sup>\[*k*+1\]</sup>,*ψ*<sup>\[*k*+1\]</sup>,*Σ*<sup>\[*k*+1\]</sup>,*μ*<sup>\[*k*+1\]</sup>,*γ*<sup>\[*k*+1\]</sup>)
are compared with the previous estimates
(*λ*<sup>\[*k*\]</sup>,*χ*<sup>\[*k*\]</sup>,*ψ*<sup>\[*k*\]</sup>,*Σ*<sup>\[*k*\]</sup>,*μ*<sup>\[*k*\]</sup>,*γ*<sup>\[*k*\]</sup>),
and it is called MCECM algorithm. The iteration will stop if the
relative increment of log-likelihood is trivial. For the online-MCECM
algorithm, after updating the
*Σ*<sup>\[*k*+1\]</sup>, *μ*<sup>\[*k*+1\]</sup>, *γ*<sup>\[*k*+1\]</sup>),
we will update the
(*δ*<sup>\[*k*+1\]</sup>,*η*<sup>\[*k*+1\]</sup>,*ζ*<sup>\[*k*+1\]</sup>),
thus, start the next step. It makes the speed of calibration to be
faster than the original one that will show in following section.

