# Attacking the Off-Policy Problem With Duality

[Link to talk](https://simons.berkeley.edu/talks/tbd-221)

There are two main challenges to off-policy RL:
* Due to lack of explicit knowledge, correcting the *distribution shift* between on-policy and off-policy state-action distributions is difficult;
* Limited data can exacerbate *extrapolation and generalization* issues in standard algorithms;

This works approaches off-policy RL via convex duality and shows that:
* The *distribution shift* problem can be attacked by *regularizing dual variables*;
* The *generalization* problem can be attacked by *regularizing primal variables*;

## Reinforcement Learning and Convex Duality
Policy evaluation can be written as a linear program (LP) as follows:

<img src="https://i.upmath.me/svg/%5Crho(%5Cpi)%20%3D%20%5Cmin_Q%20(1%20-%20%5Cgamma)%20E_%7Bs_0%20%5Csim%20%5Cmu_0%2C%20a_0%20%5Csim%20%5Cpi(s_0)%7D%5BQ(s_0%2C%20a_0)%5D" alt="\rho(\pi) = \min_Q (1 - \gamma) E_{s_0 \sim \mu_0, a_0 \sim \pi(s_0)}[Q(s_0, a_0)]" />

<img src="https://i.upmath.me/svg/%5Ctext%7Bs.t.%7D%20%5C%20Q(s%2Ca)%20%5Cgeq%20R(s%2Ca)%20%2B%20%5Cgamma%20%5Cmathcal%7BP%7D%5E%5Cpi%20Q(s%2Ca)%2C%20%5Cforall%20(s%2Ca)%20%5Cin%20S%20%5Ctimes%20A%2C" alt="\text{s.t.} \ Q(s,a) \geq R(s,a) + \gamma \mathcal{P}^\pi Q(s,a), \forall (s,a) \in S \times A," />

where <img src="https://i.upmath.me/svg/%5Crho(%5Cpi)" alt="\rho(\pi)" /> is the policy value of <img src="https://i.upmath.me/svg/%5Cpi" alt="\pi" /> and <img src="https://i.upmath.me/svg/%5Cmathcal%7BP%7D%5E%5Cpi" alt="\mathcal{P}^\pi" /> is the Bellman operator. By optimizing this LP, we push the inequalities into equalities. If the equalities are satisfied, then we have computed <img src="https://i.upmath.me/svg/Q*" alt="Q*" /> per definition. This primal version has <img src="https://i.upmath.me/svg/Q%5E%5Cpi(s%2Ca)" alt="Q^\pi(s,a)" /> as variables. The dual of the problem above can be written as:

<img src="https://i.upmath.me/svg/%5Crho(%5Cpi)%20%3D%20%5Cmax_%7Bd%20%5Cgeq%200%7D%20%5Csum_%7Bs%2Ca%7D%20d(s%2Ca)%20R(s%2Ca)" alt="\rho(\pi) = \max_{d \geq 0} \sum_{s,a} d(s,a) R(s,a)" />

<img src="https://i.upmath.me/svg/%5Ctext%7Bs.t.%7D%20%5C%20d(s%2Ca)%20%3D%20(1%20-%20%5Cgamma)%20%5Cmu_0(s)%20%5Cpi(a%7Cs)%20%2B%20%5Cmathcal%7BP%7D_*%5E%5Cpi%20d(s%2Ca)%2C%20%5Cforall%20(s%2Ca)%20%5Cin%20S%20%5Ctimes%20A%2C" alt="\text{s.t.} \ d(s,a) = (1 - \gamma) \mu_0(s) \pi(a|s) + \mathcal{P}_*^\pi d(s,a), \forall (s,a) \in S \times A," />

where <img src="https://i.upmath.me/svg/%5Cmathcal%7BP%7D_*%5E%5Cpi" alt="\mathcal{P}_*^\pi" /> is the transpose Bellman operator. The dual variables are distributions over the state-action pairs. By optimizing this LP, we obtain <img src="https://i.upmath.me/svg/d%5E*" alt="d^*" />, the on-policy distribution.

However, these formulations have too many constraints, hard to handle in stochastic and offline settings. **Convex duality** allows us to avoid these intractable constraints by applying convex regularizers. It is essential to choose the correct regularizers for this purpose.

### Regularizing the Dual

The <img src="https://i.upmath.me/svg/f" alt="f" />-divergence regularizer works well on the dual variables. The resulting formulation can be written as follows:

<img src="https://i.upmath.me/svg/%5Crho(%5Cpi)%20-%20D_f(d%5E%5Cpi%7C%7Cd%5E%7B%5Cmathcal%7BD%7D%7D)%3D%20%5Cmax_d%20-D_f(d%7C%7Cd%5E%7B%5Cmathcal%7BD%7D%7D)%20%2B%20%5Csum_%7Bs%2Ca%7D%20d(s%2Ca)%20R(s%2Ca)" alt="\rho(\pi) - D_f(d^\pi||d^{\mathcal{D}})= \max_d -D_f(d||d^{\mathcal{D}}) + \sum_{s,a} d(s,a) R(s,a)" />

<img src="https://i.upmath.me/svg/%5Ctext%7Bs.t.%7D%20%5C%20d(s%2Ca)%20%3D%20(1%20-%20%5Cgamma)%20%5Cmu_0(s)%20%5Cpi(a%7Cs)%20%2B%20%5Cmathcal%7BP%7D_*%5E%5Cpi%20d(s%2Ca)%2C%20%5Cforall%20(s%2Ca)%20%5Cin%20S%20%5Ctimes%20A%2C" alt="\text{s.t.} \ d(s,a) = (1 - \gamma) \mu_0(s) \pi(a|s) + \mathcal{P}_*^\pi d(s,a), \forall (s,a) \in S \times A," />

where <img src="https://i.upmath.me/svg/d%5E%7B%5Cmathcal%7BD%7D%7D" alt="d^{\mathcal{D}}" /> is the distribution of the offline dataset. The optimal value <img src="https://i.upmath.me/svg/d%5E*" alt="d^*" /> is still equal to <img src="https://i.upmath.me/svg/d%5E%5Cpi" alt="d^\pi" />. Now we take the convex dual of this LP and obtain the following optimization problem:

<img src="https://i.upmath.me/svg/%5Cmin_Q%20(1%20-%20%5Cgamma)%20E_%7Bs_0%20%5Csim%20%5Cmu_0%2C%20a_0%20%5Csim%20%5Cpi(s_0)%7D%5BQ(s_0%2Ca_0)%5D%20%2B%20E_%7B(s%2Ca)%20%5Csim%20d%5E%7B%5Cmathcal%7BD%7D%7D%7D%5Bf_*(R(s%2Ca)%2B%5Cgamma%5Cmathcal%7BP%7D%5E%5Cpi%20Q(s%2Ca)%20-%20Q(s%2Ca))%5D%2C" alt="\min_Q (1 - \gamma) E_{s_0 \sim \mu_0, a_0 \sim \pi(s_0)}[Q(s_0,a_0)] + E_{(s,a) \sim d^{\mathcal{D}}}[f_*(R(s,a)+\gamma\mathcal{P}^\pi Q(s,a) - Q(s,a))]," />

where <img src="https://i.upmath.me/svg/f_*" alt="f_*" /> is the conjugate of the <img src="https://i.upmath.me/svg/f" alt="f" /> chosen for the regularizer, this formulation has penalties corresponding to the convex primal's constraints. We also have an objective that includes an expected value over off-policy data.

From this, we can derive a regularized policy optimization algorithm via max-min, which wields:

<img src="https://i.upmath.me/svg/%5Cmax_%5Cpi%5Cmin_Q%20(1%20-%20%5Cgamma)%20E_%7Bs_0%20%5Csim%20%5Cmu_0%2C%20a_0%20%5Csim%20%5Cpi(s_0)%7D%5BQ(s_0%2Ca_0)%5D%20%2B%20E_%7B(s%2Ca)%20%5Csim%20d%5E%7B%5Cmathcal%7BD%7D%7D%7D%5Bf_*(R(s%2Ca)%2B%5Cgamma%5Cmathcal%7BP%7D%5E%5Cpi%20Q(s%2Ca)%20-%20Q(s%2Ca))%5D." alt="\max_\pi\min_Q (1 - \gamma) E_{s_0 \sim \mu_0, a_0 \sim \pi(s_0)}[Q(s_0,a_0)] + E_{(s,a) \sim d^{\mathcal{D}}}[f_*(R(s,a)+\gamma\mathcal{P}^\pi Q(s,a) - Q(s,a))]." />

This approach is similar to Q-Learning/Actor-Critic in the penalty term, with the difference that we take the expectation with respect to the off-policy dataset distribution. Additionally, the gradient of the inner objective with respect to the policy <img src="https://i.upmath.me/svg/%5Cpi" alt="\pi" /> wields the following term:

<img src="https://i.upmath.me/svg/d%5E%7B%5Cmathcal%7BD%7D%7D(s%2Ca)%20f_*'(R(s%2Ca)%2B%5Cgamma%5Cmathcal%7BP%7D%5E%5Cpi%20Q%5E*(s%2Ca)%20-%20Q%5E*(s%2Ca))%2C" alt="d^{\mathcal{D}}(s,a) f_*'(R(s,a)+\gamma\mathcal{P}^\pi Q^*(s,a) - Q^*(s,a))," />

which, by convex duality, is equal to <img src="https://i.upmath.me/svg/d%5E*(s%2Ca)%20%3D%20d%5E%5Cpi(s%2Ca)" alt="d^*(s,a) = d^\pi(s,a)" />. This means, when calculating gradients, the off-policy expectation is implicitly turned into a on-policy expectation.

### Regularizing the Primal

Let's go back to the primal formulation of policy evaluation. If a constraint is missing for a pair <img src="https://i.upmath.me/svg/(s%2Ca)" alt="(s,a)" />, the corresponding Q value is pushed down indefinitely and <img src="https://i.upmath.me/svg/%5Crho(%5Cpi)%20%5Crightarrow%20-%20%5Cinfty" alt="\rho(\pi) \rightarrow - \infty" />. Since our objective is to eliminate constraints, this is undesirable. This problem can be mitigated by constraining the primal to some function class <img src="https://i.upmath.me/svg/%5Cmathcal%7BF%7D" alt="\mathcal{F}" />. The authors take <img src="https://i.upmath.me/svg/%5Cmathcal%7BF%7D%20%3A%3D%20%5C%7BQ%20%5Cin%20%5Ctext%7BRKHS%2C%20s.t.%20%7D%5ClVert%20Q%20%5CrVert_%7B%5Cmathcal%7BH%7D%20%5Cleq%201%7D%5C%7D" alt="\mathcal{F} := \{Q \in \text{RKHS, s.t. }\lVert Q \rVert_{\mathcal{H} \leq 1}\}" />, that is, the unit ball in some Reproducing Kernel Hilbert Space (RKHS). The regularized version of the primal can be written as:

<img src="https://i.upmath.me/svg/%5Crho(%5Cpi)%20%3D%20%5Cmin_Q%20(1%20-%20%5Cgamma)%20E_%7Bs_0%20%5Csim%20%5Cmu_0%2C%20a_0%20%5Csim%20%5Cpi(s_0)%7D%5BQ(s_0%2C%20a_0)%5D%20%2B%20%5Cdelta_%7B%5ClVert.%5CrVert_%5Cmathcal%7BH%7D%20%5Cleq%201%7D%20(Q)" alt="\rho(\pi) = \min_Q (1 - \gamma) E_{s_0 \sim \mu_0, a_0 \sim \pi(s_0)}[Q(s_0, a_0)] + \delta_{\lVert.\rVert_\mathcal{H} \leq 1} (Q)" />

<img src="https://i.upmath.me/svg/%5Ctext%7Bs.t.%7D%20%5C%20Q(s%2Ca)%20%5Cgeq%20R(s%2Ca)%20%2B%20%5Cgamma%20%5Cmathcal%7BP%7D%5E%5Cpi%20Q(s%2Ca)%2C%20%5Cforall%20(s%2Ca)%20%5Cin%20S%20%5Ctimes%20A." alt="\text{s.t.} \ Q(s,a) \geq R(s,a) + \gamma \mathcal{P}^\pi Q(s,a), \forall (s,a) \in S \times A." />

Once again, we apply convex duality to transform constraints into penalties. The resulting optimization problem is:

<img src="https://i.upmath.me/svg/%5Cmax_%7Bd%20%5Cgeq%200%7D%20%5Clangle%20d%2CR%20%5Crangle%20-%20%5ClVert%20d%20-%20(1-%5Cgamma)%5Cmu_0%5Cpi%20-%20%5Cgamma%20%5Cmathcal%7BP%7D_*%5E%5Cpi%20d%20%5CrVert_%5Cmathcal%7BH%7D" alt="\max_{d \geq 0} \langle d,R \rangle - \lVert d - (1-\gamma)\mu_0\pi - \gamma \mathcal{P}_*^\pi d \rVert_\mathcal{H}" />

Because we chose the RKHS regularization, we can apply the *kernel trick* to expand the penalty term to:

<img src="https://i.upmath.me/svg/%5Cleft(E_%7B(s%2Ca)%20%5Csim%20d%2C%20(s'%2C%20a')%20%5Csim%20d%7D%5Bk(s%2Ca%2Cs'%2Ca')%5D%20-%202E_%7B(s%2Ca)%20%5Csim%20d%2C%20(s'%2C%20a')%20%5Csim%20%5Cmathcal%7BB%7D_8%5E%5Cpi%20d%7D%5Bk(s%2Ca%2Cs'%2Ca')%5D%20%2B%20E_%7B(s%2Ca)%20%5Csim%20%5Cmathcal%7BB%7D_8%5E%5Cpi%20d%2C%20(s'%2C%20a')%20%5Csim%20%5Cmathcal%7BB%7D_8%5E%5Cpi%20d%7D%5Bk(s%2Ca%2Cs'%2Ca')%5D%5Cright)%5E%7B%5Cfrac%7B1%7D%7B2%7D%7D%2C" alt="\left(E_{(s,a) \sim d, (s', a') \sim d}[k(s,a,s',a')] - 2E_{(s,a) \sim d, (s', a') \sim \mathcal{B}_8^\pi d}[k(s,a,s',a')] + E_{(s,a) \sim \mathcal{B}_8^\pi d, (s', a') \sim \mathcal{B}_8^\pi d}[k(s,a,s',a')]\right)^{\frac{1}{2}}," />

where <img src="https://i.upmath.me/svg/k" alt="k" /> is the kernel of the RKHS. There is no clear optimal choice for <img src="https://i.upmath.me/svg/k" alt="k" /> (and the RKHS, by extension). One such example is the energy distance, which implicitly constraints Q values to be smooth, especially when data is missing.

To make this objective off-policy, a change of variables can be done <img src="https://i.upmath.me/svg/%5Czeta(s%2Ca)%20%3D%20d(s%2Ca)%2Fd%5E%5Cmathcal%7BD%7D(s%2Ca)" alt="\zeta(s,a) = d(s,a)/d^\mathcal{D}(s,a)" />, which makes sure the expectations of objective are taken with respect to the off-policy distribution <img src="https://i.upmath.me/svg/d%5E%5Cmathcal%7BD%7D" alt="d^\mathcal{D}" />.

### Future work

The author suggests that many possible convex regularizers can be used and that there may be currently overlooked options. He also adds that there is still work to be done before applying these techniques to large-scale problems.
