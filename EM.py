import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, fsolve
from scipy.special import kv
from scipy.special import gamma as gam
from scipy.special import digamma, gammaln
from scipy.linalg import inv, pinv, det
import datetime as dt
from scipy.stats import multivariate_normal

# Plot
import matplotlib.pyplot as plt

# Derivative of the Modified Bessel Function of the third kind respect to the order
def Derivative_Bessel_Kv_lambda_(lambda_, gamma):
    h_ = 0.0001
    def obj(h):
        term1 = -kv(lambda_ + (2 * h), gamma) + 8 * kv(lambda_ + h, gamma) - 8 * kv(lambda_ - h, gamma) + kv(lambda_ - (2 * h), gamma)
        term2 = 12 * h
        return term1 / term2
    return (16 * obj(h_) - obj(2 * h_)) / 15

# Derivative of a general function respect to the order
def Derivative_General_Function(func, x, h_):
    def obj(h):
        term1 = func(x + (2 * h)) - 8 * func(x + h) + 8 * func(x - h) - func(x - (2 * h))
        term2 = 12 * h
        return term1 / term2
    return (16 * obj(h_) - obj(2 * h_)) / 15

class mGH:
    """
    Probability Density Function (PDF) of mGH distributions and its log-likelihood function.
    ==========================================================================================
    Description
    -----------
    1.First, we define the PDF of the mGH distribution.
    2.Then, we define the log-likelihood function of the mGH distribution.
    3.The Normal Inverse Gaussian's PDF is a special case of the mGH distribution.
    4.The log-likelihood function of the Normal Inverse Gaussian distribution is also defined.
    5.The Variance Gamma's PDF is a special case of the mGH distribution.
    6.The log-likelihood function of the Variance Gamma distribution is also defined.
    7.The Skewed t distribution's PDF is a special case of the mGH distribution.
    8.The log-likelihood function of the Skewed t distribution is also defined.
    """
    def __init__(self, data):
        self.x = data

        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        try:
            self.n = len(data)
            self.d = len(data[0])
        except:
            self.n = len(data)
            self.d = 1
        
        if self.d == 1:
            raise ValueError("Data must be a matrix")
        

    def GH_pdf(self, *params):
        if len(params[0]) != 6:
            print("Length of the parameters is not equal to 6", len(params[0]))
            raise ValueError("The GH distribution requires 6 parameters")
        
        lambda_, a, b, mu, Sigma, gamma = params[0]
        d = self.d
        x = self.x
        Sigma_inv = np.linalg.inv(Sigma)
        Q = b + np.inner(gamma, np.inner(Sigma_inv, gamma))
        term_c_GH_numerator = pow(np.sqrt(a * b), -lambda_) * pow(b, lambda_) * pow(Q, (d / 2) - lambda_)
        term_c_GH_denominator = pow(2 * np.pi, d / 2) * np.sqrt(np.linalg.det(Sigma)) * kv(lambda_, np.sqrt(a * b))
        c_GH = term_c_GH_numerator / term_c_GH_denominator

        rho = np.dot((x - mu), np.inner(Sigma_inv, (x - mu)))
        P = np.diag(rho) + a
        power_exp = np.inner((x - mu), np.inner(Sigma_inv, gamma))

        term_GH_numerator = kv(lambda_ - (d/2), np.sqrt(P * Q))
        term_GH_denominator = pow(np.sqrt(P *Q), (d/2) - lambda_)

        term_GH_end = np.exp(power_exp)

        GH_prob = c_GH * term_GH_numerator * term_GH_end / term_GH_denominator

        return GH_prob
    def log_likelihood_function_GH(self, *params):
        Pdf = self.GH_pdf(*params)
        return np.sum(np.log(Pdf * 1e-8))
    
    def Skewed_t_pdf(self, *params):
        if len(params[0]) != 4:
            print("Length of the parameters is not equal to 4", len(params[0]))
            raise ValueError("The Skew-t distribution requires 4 parameters")
        mu, Sigma, gamma, nu = params[0]
        d = self.d
        x = self.x
        Sigma_inv = np.linalg.inv(Sigma)
        Q = np.inner(gamma, np.inner(Sigma_inv, gamma))
        term_c_Skew_numerator = pow(2, 1 - ((nu + d) / 2) ) 
        term_c_Skew_denominator = gam(nu / 2) * pow(np.pi * nu, d / 2) * pow(np.linalg.det(Sigma), 0.5)

        c_Skew = term_c_Skew_numerator / term_c_Skew_denominator

        p_ = []
        power_exp = []
        for i in range(self.n):
            p_.append(np.inner((x[i] - mu), np.inner(Sigma_inv, (x[i] - mu))))
            power_exp.append(np.inner((x[i] - mu), np.inner(Sigma_inv, gamma)))

        P = np.array(p_)
        power_exp = np.array(power_exp)

        term_Skew_numerator = kv((nu + d) / 2, np.sqrt((P + nu) * Q)) 
        term_Skew_denominator = pow(np.sqrt((P + nu) * Q), - (nu + d) / 2)

        term_end_Skew_numerator = np.exp(power_exp)
        term_end_Skew_denominator = pow(1 + P /nu, (nu + d) / 2)

        Skew_prob = c_Skew * (term_Skew_numerator * term_end_Skew_numerator) / (term_Skew_denominator * term_end_Skew_denominator)

        return Skew_prob
    
    def log_likelihood_function_Skew(self, *params):
        Pdf = self.Skewed_t_pdf(*params)
        return np.sum(np.log(Pdf * 1e-8)) #  * 1e-8
    
    def NIG_pdf(self, *params):
        if len(params[0]) != 5:
            print("Length of the parameters is not equal to 5", len(params[0]))
            raise ValueError("The NIG distribution requires 5 parameters")
        a, b, mu, gamma, Sigma = params[0]
        d = self.d
        x = self.x
        lambda_ = -0.5
        Sigma_inv = np.linalg.inv(Sigma)
        Q = b + np.inner(gamma, np.inner(Sigma_inv, gamma))
        term_c_NIG_numerator = pow(np.sqrt(a * b), -lambda_) * pow(b, lambda_) * pow(Q, (d / 2) - lambda_)
        term_c_NIG_denominator = pow(2 * np.pi, d / 2) * np.sqrt(np.linalg.det(Sigma)) * kv(lambda_, np.sqrt(a * b))
        c_NIG = term_c_NIG_numerator / term_c_NIG_denominator
        p_ = []
        power_exp = []
        for i in range(self.n):
            p_.append(a + np.inner((x[i] - mu), np.inner(Sigma_inv, (x[i] - mu))))
            power_exp.append(np.inner((x[i] - mu), np.inner(Sigma_inv, gamma)))

        P = np.array(p_)
        power_exp = np.array(power_exp)

        term_NIG_numerator = kv(lambda_ - (d/2), np.sqrt(P * Q))
        term_NIG_denominator = pow(np.sqrt(P *Q), (d/2) - lambda_)

        term_GH_end = np.exp(power_exp - np.sqrt(a * b))

        NIG_prob = c_NIG * term_NIG_numerator * term_GH_end / term_NIG_denominator

        return NIG_prob
    def log_likelihood_function_NIG(self, *params):
        Pdf = self.NIG_pdf(*params)
        return np.sum(np.log(Pdf * 1e-8))
    
    def VG_pdf(self, *params):
        if len(params[0]) != 5:
            print("Length of the parameters is not equal to 5", len(params[0]))
            raise ValueError("The VG distribution requires 5 parameters")
        
        lambda_, b, mu, gamma, Sigma = params[0]
        d = self.d
        x = self.x
        Sigma_inv = np.linalg.inv(Sigma)
        Q = b + np.inner(gamma, np.inner(Sigma_inv, gamma))
        term_c_VG_numerator = pow(b, lambda_ / 2) * pow(Q, (d / 2) - lambda_)
        term_c_VG_denominator = pow(2 * np.pi, d / 2) * np.sqrt(np.linalg.det(Sigma)) * gam(lambda_) * pow(2, lambda_ - 1)
        c_VG = term_c_VG_numerator / term_c_VG_denominator
        p_ = []
        power_exp = []
        for i in range(self.n):
            p_.append(np.inner((x[i] - mu), np.inner(Sigma_inv, (x[i] - mu))))
            power_exp.append(np.inner((x[i] - mu), np.inner(Sigma_inv, gamma)))

        P = np.array(p_)
        power_exp = np.array(power_exp)

        term_VG_numerator = kv(lambda_ - (d/2), np.sqrt(P * Q))
        term_VG_denominator = pow(np.sqrt(P *Q), (d/2) - lambda_)

        term_VG_end = np.exp(power_exp)

        VG_prob = c_VG * term_VG_numerator * term_VG_end / term_VG_denominator

        return VG_prob
    def log_likelihood_function_VG(self, *params):
        Pdf = self.VG_pdf(*params)
        return np.sum(np.log(Pdf * 1e-8))
    

class EM_algorithm:
    r"""
    Multivariate Generalized Hyperbolic Distribution
    ================================================
    Description:
    ------------
    1. The mGH distribution is 
    X ~ mGH(\lambda, a, b, \mu, \Sigma, \gamma)
    X = \gamma Z + \mu + \sqrt{Z} A N_n
    where Z ~ GIG(\lambda, a, b), N_n is a n-dimensional standard normal random variable, and A is a n x n matrix.
    A A^T = \Sigma. 
    2. The density of the mGH distribution is
    f(x;\lambda, a, b, \mu, \sigma, \gamma) = 
    \frac{(\sqrt{ab})^{-\lambda} b^{\lambda} \left(b + \frac{\gamma^2}{\sigma^2} \right)^{\frac{1}{2} - 
    \lambda}}{\sqrt{2\pi} \sigma K_{\lambda}(\sqrt{ab})} \frac{K_{\lambda - 
    \frac{1}{2}}\left(\sqrt{\left(a + \frac{(x-\mu)^2}{\sigma^2} \right) 
    \left(b + \frac{\gamma^2}{\sigma^2} \right)} \right)}
    {\left(\sqrt{\left(a + \frac{(x-\mu)^2}{\sigma^2} \right) \left(b + \frac{\gamma^2}{\sigma^2} \right)} \right)^{\frac{1}{2} - 
    \lambda}} e^{\frac{(x-\mu)\gamma}{\sigma^2}}

    3. Using the EM algorithm to estimate the parameters of the mGH distribution.(Some Data may not be available sometimes)
    4. For now, we only consider the GH case. (In the future, we will consider the limit cases of GH, such as Normal, Laplace, etc.)
    """
    def __init__(self, data, n_iter=2000, tol=1e-20):
        self.x = data
        self.n = len(data)
        self.n_iter = n_iter
        self.tol = tol
        self.d = len(data[0])
        if self.d <= 1:
            raise ValueError("The dimension of the data should be greater than 1.")
        
    def initialize(self):
        self.x_bar = np.mean(self.x, axis=0)
        self.mu = np.mean(self.x, axis=0)
        self.Sigma = np.eye(self.d)
        self.Sample_Sigma = np.cov(self.x.T)
        self.gamma = np.ones(self.d) * 0.01
        self.a = 1
        self.b = 1
        self.lambda_ = 1
        self.theta = 1
        self.v = 2.1

    def gamma_(self):
        eta_reshaped = self.eta.reshape(-1, 1)
        gamma_inner = (self.x_bar - self.mu)
        self.gamma = gamma_inner / np.mean(self.eta)
    def gamma_skew_t(self):
        eta_reshaped = self.eta.reshape(-1, 1)
        gamma_inner = (self.x_bar - self.mu)
        self.gamma = gamma_inner / np.mean(self.eta)
    def mu_(self):
        delta_reshaped = self.delta.reshape(-1, 1)
        mu_inner = np.sum(self.x * delta_reshaped, axis=0)
        mu = ( (mu_inner / self.n - self.gamma ) / (np.mean(self.delta)) )
        self.mu = mu


    def sigma_GH(self):
        delta_reshaped = self.delta.reshape(-1, 1)
        Sigma_inner = (self.x - self.mu).T @ ((self.x - self.mu) * delta_reshaped)
        # Sigma_inner2 = np.mean([np.outer(self.x[i] - self.mu, self.gamma) for i in range(self.n)], axis=0)
        Sigma_in = (Sigma_inner / self.n) - np.mean(self.eta) * (np.outer(self.gamma, self.gamma)) #- Sigma_inner2
        if np.linalg.det(self.Sample_Sigma) == 0:
            Sample_Sigma_inv = sys.float_info.min
            Sigma = (pow(Sample_Sigma_inv, 1/self.d) * Sigma_in) / pow(np.linalg.det(Sigma_in), (1 / self.d))
        else:
            Sigma = (pow(np.linalg.det(self.Sample_Sigma), 1/self.d) * Sigma_in) / pow(np.linalg.det(Sigma_in), (1 / self.d))
        self.Sigma = Sigma
    def sigma_mu_GH(self):
        delta_reshaped = self.delta.reshape(-1, 1)
        Sigma_inner = (self.x).T @ ((self.x) * delta_reshaped)
        # Sigma_inner2 = np.mean([np.outer(self.x[i] - self.mu, self.gamma) for i in range(self.n)], axis=0)
        Sigma_in = (Sigma_inner / self.n) - np.mean(self.eta) * (np.outer(self.gamma, self.gamma)) #- Sigma_inner2
        if np.linalg.det(self.Sample_Sigma) == 0:
            Sample_Sigma_inv = sys.float_info.min
            Sigma = (pow(Sample_Sigma_inv, 1/self.d) * Sigma_in) / pow(np.linalg.det(Sigma_in), (1 / self.d))
        else:
            Sigma = (pow(np.linalg.det(self.Sample_Sigma), 1/self.d) * Sigma_in) / pow(np.linalg.det(Sigma_in), (1 / self.d))
        self.Sigma = Sigma
    def sigma_skew_t(self):
        delta_reshaped = self.delta.reshape(-1, 1)
        Sigma_inner = (self.x - self.mu).T @ ((self.x - self.mu) * delta_reshaped)
        # Sigma_inner2 = np.mean([np.outer(self.x[i] - self.mu, self.gamma) for i in range(self.n)], axis=0)
        Sigma_in = (Sigma_inner / self.n) - np.mean(self.eta) * (np.outer(self.gamma, self.gamma)) #- Sigma_inner2
        self.Sigma = Sigma_in

    def sigma_(self):
        delta_reshaped = self.delta.reshape(-1, 1)
        Sigma_inner = (self.x - self.mu).T @ ((self.x - self.mu) * delta_reshaped)
        # Sigma_inner2 = np.mean([np.outer(self.x[i] - self.mu, self.gamma) for i in range(self.n)], axis=0)
        Sigma_in = (Sigma_inner / self.n) - (np.mean(self.eta) * (np.outer(self.gamma, self.gamma))) #- Sigma_inner2
        self.Sigma = Sigma_in
    def sigma_mu(self):
        delta_reshaped = self.delta.reshape(-1, 1)
        Sigma_inner = (self.x).T @ ((self.x) * delta_reshaped)
        # Sigma_inner2 = np.mean([np.outer(self.x[i] - self.mu, self.gamma) for i in range(self.n)], axis=0)
        Sigma_in = (Sigma_inner / self.n) - (np.mean(self.eta) * (np.outer(self.gamma, self.gamma))) #- Sigma_inner2
        self.Sigma = Sigma_in
    def Q_P(self, a, b):
        try:
            # 尝试直接使用 np.linalg.inv 并添加小的正值到对角线上以提高稳定性
            Sigma_inv = inv(self.Sigma)
        except:
            print("LinAlgError: SVD did not converge when inverting Sigma.")
            # 处理异常，例如使用 pinv 作为备选方案或设置 Sigma_inv 为单位矩阵等
            Sigma_inv = pinv(self.Sigma)  # 或其他适当的处理方式
        rho = np.sum(((self.x - self.mu) @ Sigma_inv) * (self.x - self.mu), axis=1)
        P_val = (rho + a) / (b + self.gamma @ Sigma_inv @ self.gamma)
        Q_val = (rho + a) * (b + self.gamma @ Sigma_inv @ self.gamma)
        self.Q = Q_val
        self.P = P_val
    
    def Eta(self):
        self.eta = (np.sqrt(self.P) * kv(self.lambda_ - (self.d / 2) + 1, np.sqrt(self.Q))) \
            / kv(self.lambda_ - (self.d / 2), np.sqrt(self.Q))
        
    def Delta(self):
        self.delta = (kv(self.lambda_ - (self.d / 2) - 1, np.sqrt(self.Q)))\
            / (kv(self.lambda_ - (self.d / 2), np.sqrt(self.Q)) * np.sqrt(self.P))
    
    def Xi(self):
        derivative_alpha = Derivative_Bessel_Kv_lambda_(self.lambda_ - self.d / 2, np.sqrt(self.Q))
        self.xi = (1 / 2) * np.log(self.P) + (derivative_alpha / kv(self.lambda_ - self.d / 2, np.sqrt(self.Q)))

    def Theta_func(self, theta):
        term1 = pow(self.n, 2) * kv(self.lambda_ - 1, theta) * kv(self.lambda_ + 1, theta)
        term2 = np.sum(self.delta) * np.sum(self.eta) * kv(self.lambda_, theta) * kv(self.lambda_, theta)
        return (term1 - term2) ** 2

    
    def lambda_func(self, lambda_):
        inner = np.sqrt(self.a * self.b)
        mean_xi = np.mean(self.xi)
        derivative_lambda_Bessel = Derivative_Bessel_Kv_lambda_(lambda_, inner)
        bessel_lambda_ab = kv(lambda_, inner)
        term = mean_xi - (0.5 * np.log(self.a)) + (0.5 * np.log(self.b)) - (derivative_lambda_Bessel / bessel_lambda_ab)
        return term ** 2

    def a_b_GH_update(self):
        self.theta = minimize(self.Theta_func, self.theta, method='Nelder-Mead').x
        epsi = kv(self.lambda_ - 1, self.theta) / (np.mean(self.delta) * kv(self.lambda_, self.theta))
        # Find a and b
        b = (self.theta) / epsi

        a = np.power(self.theta, 2) / b
        self.a = a
        self.b = b
    
    def lambda_update_GH(self):
        # Minimize the lambda function
        lambda_optimization_result = minimize(self.lambda_func, self.lambda_, method='Nelder-Mead')
        self.lambda_ = lambda_optimization_result.x

    # EM algorithm for the GH distribution+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def GH_estimate(self):
        # E step
        self.Q_P(self.a, self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        # M step
        self.gamma_()
        self.mu_()
        self.sigma_GH()
        self.a_b_GH_update()
        self.lambda_update_GH()
    
    def GH_mu_estimate(self):
        self.mu = np.zeros(self.d)
        # E step
        self.Q_P(self.a, self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        # M step
        self.gamma_()
        self.sigma_mu_GH()
        self.a_b_GH_update()
        self.lambda_update_GH()
    
    def GH_estimate_online(self):
        # E step
        self.Q_P(self.a, self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        # M step
        self.gamma_()
        self.mu_()
        self.sigma_GH()
        # Online
        self.gamma_()
        self.mu_()
        self.sigma_GH()

        self.a_b_GH_update()
        self.lambda_update_GH()

    def Q_P_skew_t(self, v):
        try:
            # 尝试直接使用 np.linalg.inv 并添加小的正值到对角线上以提高稳定性
            Sigma_inv = inv(self.Sigma)
        except:
            print("LinAlgError: SVD did not converge when inverting Sigma.")
            # 处理异常，例如使用 pinv 作为备选方案或设置 Sigma_inv 为单位矩阵等
            Sigma_inv = pinv(self.Sigma)  # 或其他适当的处理方式
        rho = np.sum(((self.x - self.mu) @ Sigma_inv) * (self.x - self.mu), axis=1)
        P_val = (rho + v) / (self.gamma @ Sigma_inv @ self.gamma)
        Q_val = (rho + v) * (self.gamma @ Sigma_inv @ self.gamma)
        self.Q = Q_val
        self.P = P_val

    def Delta_skew_t(self):
        self.delta = (kv((self.d + self.v + 2) / 2 , np.sqrt(self.Q)))\
            / (kv((self.v + self.d) / 2, np.sqrt(self.Q)) * np.sqrt(self.P))

    def Eta_skew_t(self):
        self.eta = (np.sqrt(self.P) * kv((self.v + self.d - 2) / 2, np.sqrt(self.Q))) \
            / kv((self.v + self.d) / 2, np.sqrt(self.Q))
        
    def Xi_skew_t(self):
        derivative_alpha = Derivative_Bessel_Kv_lambda_(- (self.v + self.d) / 2 , np.sqrt(self.Q))
        self.xi = (1 / 2) * np.log(self.P) + (derivative_alpha / kv((self.v + self.d) / 2, np.sqrt(self.Q)))


    def v_skewt_func(self, v):
        term1 = np.log(v / 2) + 1 - np.mean(self.xi) - np.mean(self.delta) - digamma(v / 2)
        
        return term1 ** 2
    
    def v_skewt_update(self):
        # # Update the v
        v_optimization_result = minimize(self.v_skewt_func, self.v, bounds=[(2, None)], method='Nelder-Mead')
        self.v = v_optimization_result.x
        if self.v <= 2:
            self.v = 2.1
    def v_skewt_func_root(self, v):
        term1 = np.log(v / 2) - np.mean(self.xi) - np.mean(self.delta) - digamma(v / 2)
        
        return term1
    def v_skewt_root(self):
        # Find the root of the v_skewt_func
        v_root = fsolve(self.v_skewt_func_root, self.v)
        self.v = v_root[0]
        if self.v <= 2:
            self.v = 2.1

    # EM algorithm for the Skewed-t distribution+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def Skewt_estimate(self):
        # E step
        self.a = 0
        self.b = 0

        self.Q_P_skew_t(self.v)
        self.Eta_skew_t()
        self.Delta_skew_t()
        self.Xi_skew_t()

        # M step
        self.mu_()
        self.gamma_()
        self.sigma_skew_t()


        self.v_skewt_update()

    def Skewt_mu_estimate(self):
        self.mu = np.zeros(self.d)
        # E step
        self.a = 0
        self.b = 0
        self.Q_P_skew_t(self.v)
        self.Eta_skew_t()
        self.Delta_skew_t()
        self.Xi_skew_t()

        # M step
        self.gamma_()
        self.sigma_mu_GH()


        self.v_skewt_update()
    def Skewt_estimate_online(self):
        # E step
        self.a = 0
        self.b = 0
        self.Q_P_skew_t(self.v)
        self.Eta_skew_t()
        self.Delta_skew_t()
        self.Xi_skew_t()

        # M step
        self.mu_()
        self.gamma_()
        self.sigma_skew_t()
        self.mu_()
        self.gamma_()
        self.sigma_skew_t()
        

        self.v_skewt_update()
        # self.v_skewt_root()


    def a_b_NIG_update(self):
        self.theta = 1 / (np.mean(self.eta) * np.mean(self.delta) - 1)
        self.a = self.theta * np.mean(self.eta)
        self.b = np.mean(self.delta) * self.theta - (1 / np.mean(self.eta))
    # EM algorithm for the NIG distribution+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def NIG_estimate(self):
        # E step
        self.lambda_ = -0.5
        self.Q_P(self.a, self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        # M step
        self.gamma_()
        self.mu_()
        self.sigma_()
        
        self.a_b_NIG_update()

    def NIG_mu_estimate(self):
        self.mu = np.zeros(self.d)
        # E step
        self.lambda_ = -0.5
        self.Q_P(self.a, self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        # M step
        self.gamma_()
        self.sigma_mu()
        
        self.a_b_NIG_update()


    def NIG_estimate_online(self):
        # E step
        self.lambda_ = -0.5
        self.Q_P(self.a, self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        # M step
        self.gamma_()
        self.mu_()
        self.sigma_()

        # Online
        self.lambda_ = -0.5
        self.Q_P(self.a, self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        self.a_b_NIG_update()

    def Q_P_VG(self, b):
        try:
            # 尝试直接使用 np.linalg.inv 并添加小的正值到对角线上以提高稳定性
            Sigma_inv = inv(self.Sigma)
        except:
            print("LinAlgError: SVD did not converge when inverting Sigma.")
            # 处理异常，例如使用 pinv 作为备选方案或设置 Sigma_inv 为单位矩阵等
            Sigma_inv = pinv(self.Sigma)  # 或其他适当的处理方式
        rho1 = (self.x - self.mu) @ Sigma_inv @ (self.x - self.mu).T
        rho = np.diag(rho1)
        P_val = (rho) / (b + self.gamma @ Sigma_inv @ self.gamma)
        Q_val = (rho) * (b + self.gamma @ Sigma_inv @ self.gamma)
        self.Q = Q_val
        self.P = P_val

    def lambda_func_VG(self, lambda_):
        mean_eta = np.mean(self.eta)
        mean_xi = np.mean(self.xi)
        term = np.log(lambda_) - np.log(mean_eta) + mean_xi - digamma(lambda_)
        return term ** 2

    def b_lambda_VG_update(self):
        # Find lambda
        lambda_optimization_result = minimize(self.lambda_func_VG, self.lambda_, bounds=[(1e-5, None)], 
                                             tol=1e-10, options={'maxfun': 10000})
        self.lambda_ = lambda_optimization_result.x
        # Find b 
        self.b = (2 * self.lambda_) / np.mean(self.eta)
        

    # EM algorithm for the VG distribution+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def VG_estimate(self):
        # E step
        self.a = 0
        self.Q_P_VG(self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        # M step
        self.gamma_()
        self.mu_()
        self.sigma_()
        self.b_lambda_VG_update()

    def VG_estimate_online(self):
        # E step
        self.a = 0
        self.Q_P_VG(self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        # M step
        self.gamma_()
        self.mu_()
        self.sigma_()

        #Online
        self.Q_P_VG(self.b)
        self.Eta()
        self.Delta()
        self.Xi()

        self.b_lambda_VG_update()

    def fit_GH(self):
        self.initialize()
        GH_Log_Likelihood = []
        GH_Dis = mGH(self.x)
        parms = []
        if self.online:
            Esimation = self.GH_estimate_online
        else:
            Esimation = self.GH_estimate
        for i in range(self.n_iter):
            Esimation()
            parms.append([self.lambda_, self.a, self.b, self.mu, self.Sigma, self.gamma])
            log_likelihood = GH_Dis.log_likelihood_function_GH(parms[-1])
            GH_Log_Likelihood.append(log_likelihood)
            if i > 1:
                if abs(GH_Log_Likelihood[i] - GH_Log_Likelihood[i - 1]) < self.tol:
                    break
        
        most_likely_parms = parms[-1]
        return most_likely_parms, GH_Log_Likelihood
    
    def fit_GH_mu(self):
        self.initialize()
        GH_mu_Log_Likelihood = []
        GH_mu_Dis = mGH(self.x)
        parms = []
        Esimation = self.GH_mu_estimate

        for i in range(self.n_iter):
            Esimation()
            parms.append([self.lambda_, self.a, self.b, self.mu, self.Sigma, self.gamma])
            log_likelihood = GH_mu_Dis.log_likelihood_function_GH(parms[-1])
            GH_mu_Log_Likelihood.append(log_likelihood)
            if i > 1:
                if abs(GH_mu_Log_Likelihood[i] - GH_mu_Log_Likelihood[i - 1]) < self.tol:
                    break
        
        most_likely_parms = parms[-1]
        return most_likely_parms, GH_mu_Log_Likelihood
        
    def fit_Skewt(self):
        self.initialize()
        # self.mu_skew_t()
        # self.sigma_skew_t()
        # self.gamma = np.zeros(self.d)
        # self.mu = self.mu - (self.v / (self.v - 2)) * self.gamma
        # self.Sigma = (np.cov(self.x.T) * (self.v - 2) / self.v) - (2 * self.v / ((self.v - 2) *(self.v - 4)) * np.outer(self.gamma, self.gamma))
        Skewt_Log_Likelihood = []
        Skewt_Dis = mGH(self.x)
        parms = []
        if self.online:
            Esimation = self.Skewt_estimate_online
        else:
            Esimation = self.Skewt_estimate
        for i in range(self.n_iter):
            Esimation()
            parms.append([self.mu, self.Sigma, self.gamma, self.v])
            log_likelihood = Skewt_Dis.log_likelihood_function_Skew(parms[-1])
            Skewt_Log_Likelihood.append(log_likelihood)
            if i > 1:
                if abs(Skewt_Log_Likelihood[i] - Skewt_Log_Likelihood[i - 1]) < self.tol:
                    break
        
        most_likely_parms = parms[-1]
        return most_likely_parms, Skewt_Log_Likelihood
    
    def fit_Skewt_mu(self):
        self.initialize()
        Skewt_Log_Likelihood = []
        Skewt_Dis = mGH(self.x)
        parms = []
        Esimation = self.Skewt_mu_estimate
        for i in range(self.n_iter):
            Esimation()
            parms.append([self.mu, self.Sigma, self.gamma, self.v])
            log_likelihood = Skewt_Dis.log_likelihood_function_Skew(parms[-1])
            Skewt_Log_Likelihood.append(log_likelihood)
            if i > 1:
                if abs(Skewt_Log_Likelihood[i] - Skewt_Log_Likelihood[i - 1]) < self.tol:
                    break
        
        most_likely_parms = parms[-1]
        return most_likely_parms, Skewt_Log_Likelihood
    def fit_NIG(self):
        self.initialize()
        NIG_Log_Likelihood = []
        NIG_Dis = mGH(self.x)
        parms = []
        if self.online:
            Esimation = self.NIG_estimate_online
        else:
            Esimation = self.NIG_estimate
        for i in range(self.n_iter):
            Esimation()
            parms.append([self.a, self.b, self.mu, self.gamma, self.Sigma])
            log_likelihood = NIG_Dis.log_likelihood_function_NIG(parms[-1])
            NIG_Log_Likelihood.append(log_likelihood)
            if i > 1:
                if abs(NIG_Log_Likelihood[i] - NIG_Log_Likelihood[i - 1]) < self.tol:
                    break
        
        most_likely_parms = parms[-1]
        return most_likely_parms, NIG_Log_Likelihood
    
    def fit_NIG_mu(self):
        self.initialize()
        NIG_Log_Likelihood = []
        NIG_Dis = mGH(self.x)
        parms = []
        Esimation = self.NIG_mu_estimate

        for i in range(self.n_iter):
            Esimation()
            parms.append([self.a, self.b, self.mu, self.gamma, self.Sigma])
            log_likelihood = NIG_Dis.log_likelihood_function_NIG(parms[-1])
            NIG_Log_Likelihood.append(log_likelihood)
            if i > 1:
                if abs(NIG_Log_Likelihood[i] - NIG_Log_Likelihood[i - 1]) < self.tol:
                    break
        
        most_likely_parms = parms[-1]
        return most_likely_parms, NIG_Log_Likelihood
    def fit_VG(self):
        self.initialize()
        VG_Log_Likelihood = []
        VG_Dis = mGH(self.x)
        parms = []
        if self.online:
            Esimation = self.VG_estimate_online              
        else:
            Esimation = self.VG_estimate
        for i in range(self.n_iter):
            Esimation()
            parms.append([self.lambda_, self.b, self.mu, self.gamma, self.Sigma])
            log_likelihood = VG_Dis.log_likelihood_function_VG(parms[-1])
            VG_Log_Likelihood.append(log_likelihood)
            if i > 1:
                if abs(VG_Log_Likelihood[i] - VG_Log_Likelihood[i - 1]) < self.tol:
                    break
        
        most_likely_parms = parms[-1]
        return most_likely_parms, VG_Log_Likelihood
    
    def run(self, distribution, online=False):

        self.online = online
        if type(distribution) is str:
            pass
        else:
            distribution = str(distribution)

        
        if distribution == "GH":
            most_likely_parameters = self.fit_GH()
        elif distribution == "GH_mu":
            most_likely_parameters = self.fit_GH_mu()
        
        elif distribution == "Skew-t":
            most_likely_parameters = self.fit_Skewt()
        elif distribution == "Skew-t_mu":
            most_likely_parameters = self.fit_Skewt_mu()
        
        elif distribution == "NIG":
            most_likely_parameters = self.fit_NIG()
        elif distribution == "NIG_mu":
            most_likely_parameters = self.fit_NIG_mu()

        elif distribution == "VG":
            most_likely_parameters = self.fit_VG()
        
        

        else:
            raise ValueError("%s Distribution is not in our algo...")

        return most_likely_parameters
