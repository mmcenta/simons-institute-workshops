# Offline Deep Reinforcement Learning Algorithms

[Link to video](https://simons.berkeley.edu/talks/tbd-216) | [Link to paper](https://arxiv.org/abs/2006.04779)

* For this talk, we refer to the policy that collected the offline dataset as the behavior policy <img src="https://i.upmath.me/svg/%5Cpi_%5Cbeta" alt="\pi_\beta" />.

* A motivation for offline RL: to be able to leverage large datasets like supervised learning;

## Why is offline RL difficult?
* Using current baselines, there is a significant performance gap between agents trained with purely offline data and those trained on the offline data and fine-tuned for a few steps on online data.
* *Overfitting is the issue*. Experiments show little change in performance as the dataset size increases, which refutes this hypothesis. A massive overestimation of Q values is also observed in these experiments.
* *Training data is not good*. This is usually not the case, because training a behavioral cloning algorithm on the best data usually outperforms offline reinforcement learning methods.
* *Distributional shift*. Q-learning is done via empirical risk minimization of the expected value of the mean squared error between the actual Q-value and a bootstrapped target. This gives us guarantees over the performance of the estimator on the training data distribution. Still, nothing can be said about *different distributions* (such as other policies as the one used to collect the data) and *pointwise risk* (that is, there is no guarantee that the risk will be low for a given point). This problem is exaggerated in Q-learning because the target is calculated wither with a <img src="https://i.upmath.me/svg/%5Cmax" alt="\max" /> or a <img src="https://i.upmath.me/svg/%5Cargmax" alt="\argmax" /> operator, which tends to select noisy peaks in the learned Q function. This explains the observed overestimation of Q-values.

## How do we design offline RL algorithms?

Prior methods address the distributional shift issue by adding a "policy constraint" that assures that the policy used to bootstrap the target smaller KL-divergence to the behavior policy than a given <img src="https://i.upmath.me/svg/%5Cepsilon%20%3E%200" alt="\epsilon &gt; 0" />. But there are a couple of issues with this method:
  * This approach might be too conservative. This constraint forces the learned policy is close to the behavior policy, which is not necessarily good. For example, if the data is collected using a random policy, this constraint will limit how deterministic the learned policy will be, which might be a problem. This can be partially mitigated with a support constraint, as shown [here](https://arxiv.org/abs/1906.00949).
  * The behavior policy might be difficult to estimate, in which case performance is hurt as observed experimentally. There's also evidence that more powerful policy estimators result in improvements in several tasks, suggesting that estimation the behavior policy might be the bottleneck for this method.

Another solution is to avoid estimating behavior policies with implicit constraints. The Lagrangian of the constrained policy update of policy constraint models can be computed, and a solution can be expressed in function of the advantage and the Lagrange multiplier. The optimal solution can then be approximated via weighted maximum likelihood, maximizing an expected value over samples from the dataset. For more details, read the [Advantage-Weighted Regression (AWR) paper](https://arxiv.org/abs/1910.00177).

## Conservative Q-Learning

Instead of constraining the learned policy to be close to the behavioral policy, Conservative Q-Learning (CQL) aims to avoid overestimation.

To avoid the overestimation of Q values, the authors propose to add a term to the common Q-learning objective that pushes down on large Q-values. The optimization problem is written as

<img src="https://i.upmath.me/svg/%5Chat%7BQ%7D%5E%5Cpi%20%3D%20%5Carg%5Cmin_%7BQ%7D%5Cleft(%5Cmax_%5Cmu%20%5C%20%5Calpha%20E_%7Bs%20%5Csim%20D%2C%20a%20%5Csim%20%5Cmu(a%7Cs)%7D%5BQ(s%2Ca)%5D%20%2B%20E_%7B(s%2Ca%2Cs')%20%5Csim%20D%7D%20%5B(Q(s%2Ca)%20-%20(r(s%2Ca)%20%2B%20E_%5Cpi%5BQ(s'%2C%20a')%5D)%5E2%5D%5Cright)%2C" alt="\hat{Q}^\pi = \arg\min_{Q}\left(\max_\mu \ \alpha E_{s \sim D, a \sim \mu(a|s)}[Q(s,a)] + E_{(s,a,s') \sim D} [(Q(s,a) - (r(s,a) + E_\pi[Q(s', a')])^2]\right)," />

where the first term is the added constraint, and the second is the regular Q-learning objective. It can be shown that learning with this objective with a large enough <img src="https://i.upmath.me/svg/%5Calpha" alt="\alpha" /> results in <img src="https://i.upmath.me/svg/%5Chat%7BQ%7D%5E%5Cpi%20%5Cleq%20Q%5E%5Cpi" alt="\hat{Q}^\pi \leq Q^\pi" /> for all state-action pairs, that is, *all learned Q values are lower bounds*.

By adding a term that pushes up on samples from the offline dataset, a tighter bound can be obtained. The resulting optimization problem can be expressed as


<img src="https://i.upmath.me/svg/%5Chat%7BQ%7D%5E%5Cpi%20%3D%20%5Carg%5Cmin_%7BQ%7D%5Cleft(%5Cmax_%5Cmu%20%5C%20%5Calpha%20E_%7Bs%20%5Csim%20D%2C%20a%20%5Csim%20%5Cmu(a%7Cs)%7D%5BQ(s%2Ca)%5D%20-%20%5Calpha%20E_%7B(s%2Ca)%20%5Csim%20D%7D%5BQ(s%2Ca)%5D%20%2B%20E_%7B(s%2Ca%2Cs')%20%5Csim%20D%7D%20%5B(Q(s%2Ca)%20-%20(r(s%2Ca)%20%2B%20E_%5Cpi%5BQ(s'%2C%20a')%5D)%5E2%5D%5Cright)%2C" alt="\hat{Q}^\pi = \arg\min_{Q}\left(\max_\mu \ \alpha E_{s \sim D, a \sim \mu(a|s)}[Q(s,a)] - \alpha E_{(s,a) \sim D}[Q(s,a)] + E_{(s,a,s') \sim D} [(Q(s,a) - (r(s,a) + E_\pi[Q(s', a')])^2]\right)," />

where the third term is the term that pushes up on Q values with both state and action from the dataset. The intuition is that when the policy closely matches the data, the two terms have a small effect, but when the model starts overestimating Q values the first term dominates and avoids the problem. In this case, there is no guarantee that <img src="https://i.upmath.me/svg/%5Chat%7BQ%7D%5E%5Cpi%20%5Cleq%20Q%5E%5Cpi" alt="\hat{Q}^\pi \leq Q^\pi" /> for all state-action pairs, but instead that <img src="https://i.upmath.me/svg/E_%7B%5Cpi%7D%5B%5Chat%7BQ%7D%5E%5Cpi%5D%20%5Cleq%20E_%7B%5Cpi%7D%5BQ%5E%5Cpi%5D" alt="E_{\pi}[\hat{Q}^\pi] \leq E_{\pi}[Q^\pi]" />, which is sufficient to train the policy. Experimentally, this methods do indeed underestimate Q values and the bound obtained with the second formulation is observed to be closer to the actual Q values.

For evaluation, the authors used [D4RL](https://arxiv.org/abs/2004.07219), a benchmark of offline reinforcement learning with varied datasets. The experimental results show that CQL is substantially better than previous baselines.
