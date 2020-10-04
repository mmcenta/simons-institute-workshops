# Learning from the Past Without Great Exploration

[Link to video](https://simons.berkeley.edu/talks/tbd-215) | [Link to paper](https://arxiv.org/abs/2007.08202)

* Motivation: learning from limited samples to robustly make good decisions;

* Goal: offline policy optimization with generalization guarantees, that is, structural risk minimization for offline reinforcement learning;

* Goal: practical offline policy optimization with lower bounds on performance, that is, cross validation for reinforcement learning;

* The two common evaluation criteria for off policy evaluation are computational efficiency and performance accuracy;

* Most of the work up to now make strong hypotheses (see [Chen and Jiang, 2019](https://arxiv.org/abs/1905.00360) for more details). Let <img src="https://i.upmath.me/svg/%5Cmathcal%7BF%7D" alt="\mathcal{F}" /> be the set of estimators, then some of these assumptions are:
  * *Completeness*: let <img src="https://i.upmath.me/svg/%5Cmathcal%7BT%7D" alt="\mathcal{T}" /> be the Bellman backup operator, then <img src="https://i.upmath.me/svg/%5Cforall%20f%20%5Cin%20%5Cmathcal%7BF%7D%2C%20%5Cmathcal%7BT%7D%20f%20%5Cin%20%5Cmathcal%7BF%7D" alt="\forall f \in \mathcal{F}, \mathcal{T} f \in \mathcal{F}" />.
  * *Realizability*: let <img src="https://i.upmath.me/svg/Q%5E*" alt="Q^*" /> be the optimal state-action value function, then <img src="https://i.upmath.me/svg/Q%5E*%20%5Cin%20%5Cmathcal%7BF%7D" alt="Q^* \in \mathcal{F}" />.
  * *Overlap Assumption*: for every state-action pair that is reachable by the MDP, there is good support. More formally, let <img src="https://i.upmath.me/svg/%5Cmu" alt="\mu" /> be the probability distribution over state-actions of the dataset and <img src="https://i.upmath.me/svg/%5Cupsilon" alt="\upsilon" /> the probability distribution of reaching a state-action with a policy. Then, we suppose that, for all policies, <img src="https://i.upmath.me/svg/%5Cforall(s%2Ca)%20%5Cin%20%5Cmathcal%7BS%7D%20%5Ctimes%20%5Cmathcal%7BA%7D%2C%20%5Cexists%20C" alt="\forall(s,a) \in \mathcal{S} \times \mathcal{A}, \exists C" /> such that <img src="https://i.upmath.me/svg/%5Cfrac%7B%5Cupsilon(s%2C%20a)%7D%7B%5Cmu(s%2C%20a)%7D%20%5Cleq%20C" alt="\frac{\upsilon(s, a)}{\mu(s, a)} \leq C" />.

* This works concentrates on guarantees on finding the best in class policy, instead of generalization guarantees on policy performance.

* New offline reinforcement learning algorithms such as [BCQ](https://arxiv.org/abs/1812.02900) rely on the overlap assumption, but this work concentrates on off policy reinforcement learning without full data coverage.

* This paper proposes a few small tabular environments that illustrate some of the problem with current baselines. Their experiments show that the baselines fail when rare states are present, which causes overestimation of values. Because the environments are tabular, the authors implement modified versions of Policy Iteration (PI) and Value Iteration (VI).

* The method introduced by this work is to pessimistic values for the state-action space with insufficient data. They implement this via a filtration function <img src="https://i.upmath.me/svg/%5Czeta%20(s%2C%20a%3B%20%5Cmu%2C%20b)%20%3D%201_%7B%5Cmu(s%2C%20a)%20%3E%20b%7D%7D" alt="\zeta (s, a; \mu, b) = 1_{\mu(s, a) &gt; b}}" />. This function is introduced in the Bellman evaluation operator, which is equivalent to assuming zero rewards for the filtered tuples. An interpretation is that this method (implicitly) defines a class of policies with sufficient support and it tries to do the best it can within that support.

* Finally, the theoretical result are error bounds on PI and VI, which I will not reproduce here but can be found on the linked paper.
