# MOPO: Model-Based Offline Policy Optimization

[Link to video](https://simons.berkeley.edu/talks/tbd-206) | [Link to paper](https://arxiv.org/abs/2005.13239)

* In this talk, the dataset of the offline reinforcement learning is called *batch*;

* Due to distribution shift, learning on the batch only guarantees accurate predictions on the batch data distribution;

* Strong Pessimism/Conservatism: stay inside the support of the batch data distribution;

* Mild Pessimism/Conservatism: go outside the support by trading off return and risk. In offline reinforcement learning, this translates to:
  * Step 1: build uncertainty quantification (UQ) of return.

    <img src="https://i.upmath.me/svg/%5Ceta%5E*(%5Cpi)%20%5Cin%20%5B%5Chat%7B%5Ceta%7D(%5Cpi)%20%5Cpm%20e(%5Cpi)%5D" alt="\eta^*(\pi) \in [\hat{\eta}(\pi) \pm e(\pi)]" />

  * Step 2: maximize the lower confidence bound.

    <img src="https://i.upmath.me/svg/%5Cmax_%5Cpi%20%5Chat%7B%5Ceta%7D(%5Cpi)%20-%20e(%5Cpi)" alt="\max_\pi \hat{\eta}(\pi) - e(\pi)" />

## First Step: Build UQ of return

* To do this, a model-based approach is used, which uses UQ of learned dynamics to obtain UQ for the return.
  * First learn a **deterministic** dynamical model <img src="https://i.upmath.me/svg/%5Chat%7BT%7D" alt="\hat{T}" /> is learning on the offline data;
  * Assume that <img src="https://i.upmath.me/svg/V%5E%7B%5Cpi%2C%20T%5E*%7D" alt="V^{\pi, T^*}" /> is <img src="https://i.upmath.me/svg/c" alt="c" />-Lipschitz and an error estimator <img src="https://i.upmath.me/svg/u" alt="u" /> for <img src="https://i.upmath.me/svg/%5Chat%7BT%7D" alt="\hat{T}" /> satisfying
    <img src="https://i.upmath.me/svg/%5Cleft%5ClVert%20%5Chat%7BT%7D(s%2C%20a)%20-%20T%5E*(s%2C%20a)%20%5Cright%5CrVert%20%5Cleq%20u(s%2C%20a)%3B" alt="\left\lVert \hat{T}(s, a) - T^*(s, a) \right\rVert \leq u(s, a);" />
  * Then, we have the following relation between the return error estimator <img src="https://i.upmath.me/svg/e" alt="e" /> and the dynamical error estimator <img src="https://i.upmath.me/svg/u" alt="u" />:
    <img src="https://i.upmath.me/svg/e(%5Cpi)%20%3D%20%5Cfrac%7Bc%5Cgamma%7D%7B1-%5Cgamma%7DE_%7B(s%2Ca)%5Csim%5Cpi%2C%5Chat%7BT%7D%7D%5Cleft%5Bu(s%2Ca)%5Cright%5D." alt="e(\pi) = \frac{c\gamma}{1-\gamma}E_{(s,a)\sim\pi,\hat{T}}\left[u(s,a)\right]." />

* There is a set of more general assumptions that lead to the same result:
  * Assume that <img src="https://i.upmath.me/svg/V%5E%7B%5Cpi%2C%20T%5E*%7D%20%5Cin%20c%20%5Ccdot%20%5Cmathcal%7BF%7D" alt="V^{\pi, T^*} \in c \cdot \mathcal{F}" />, where <img src="https://i.upmath.me/svg/c%20%5Cin%20%5Cmathbb%7BR%7D" alt="c \in \mathbb{R}" /> and <img src="https://i.upmath.me/svg/%5Cmathcal%7BF%7D" alt="\mathcal{F}" /> is a family of functions;
  * Assume error estimator <img src="https://i.upmath.me/svg/u" alt="u" /> for the learned (**possibly stochastic**) dynamics <img src="https://i.upmath.me/svg/%5Chat%7BT%7D" alt="\hat{T}" /> satisfying <img src="https://i.upmath.me/svg/d_%7B%5Cmathcal%7BF%7D%7D%5Cleft(%5Chat%7BT%7D(s%2C%20a)%2C%20T%5E*(s%2C%20a)%5Cright)%20%5Cleq%20u(s%2C%20a)" alt="d_{\mathcal{F}}\left(\hat{T}(s, a), T^*(s, a)\right) \leq u(s, a)" />, where <img src="https://i.upmath.me/svg/d_%7B%5Cmathcal%7BF%7D%7D" alt="d_{\mathcal{F}}" /> is an integral probability metric between two distributions with respect to <img src="https://i.upmath.me/svg/%5Cmathcal%7BF%7D" alt="\mathcal{F}" />;
  * Then we have the same result;

## Second Step: Optimize lower bound with reward penalty

* We optimize <img src="https://i.upmath.me/svg/%5Chat%7B%5Ceta%7D(%5Cpi)%20-%20e(%5Cpi)%20%3D%20E_%7B(s%2Ca)%20%5Csim%20%5Cpi%2C%20%5Chat%7BT%7D%7D%20%5Cleft%5Br(s%2C%20a)%20-%20%5Clambda%20u(s%2C%20a)%5Cright%5D" alt="\hat{\eta}(\pi) - e(\pi) = E_{(s,a) \sim \pi, \hat{T}} \left[r(s, a) - \lambda u(s, a)\right]" /> by:
  * Defining an MDP <img src="https://i.upmath.me/svg/%5Ctilde%7BM%7D" alt="\tilde{M}" /> with the learned dynamics <img src="https://i.upmath.me/svg/%5Chat%7BT%7D" alt="\hat{T}" /> and penalized rewards <img src="https://i.upmath.me/svg/%5Ctilde%7Br%7D(s%2C%20a)%20%3D%20r(s%2Ca)%20-%20%5Clambda%20u(s%2C%20a)" alt="\tilde{r}(s, a) = r(s,a) - \lambda u(s, a)" />;
  * Finding the optimal policy of <img src="https://i.upmath.me/svg/%5Ctilde%7BM%7D" alt="\tilde{M}" /> with an off-the-shelf RL algorithm;
