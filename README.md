# MDPs.jl

This lightweight package primarily exports the `AbstractMDP` and `AbstractPolicy` type and the API for working with MDPs (e.g., the `step!` and `reset!` functions). It also provides an `interact` function for interacting with an MDP using a policy while supporting callbacks using hooks.

Some basic MDPs, policies, wrappers and hooks are also provided, such as `RandomDiscreteMDP`, `RandomPolicy`, `GreedyPolicy`, `FrameStackWrapper`, `EmpiricalPolicyEvaluationHook` and `ProgressMeterHook`.

See documentation on how to define your own MDPs, policies and hooks.

This package is intended to be used with other packages like [GridWorlds.jl](https://github.com/bhatiaabhinav/GridWorlds.jl), [ClassicControl.jl](https://github.com/bhatiaabhinav/ClassicControl.jl), [ValueIteration.jl](https://github.com/bhatiaabhinav/ValueIteration.jl), [TabularRL.jl](https://github.com/bhatiaabhinav/TabularRL.jl), [DQN.jl](https://github.com/bhatiaabhinav/DQN.jl), [PPO.jl](https://github.com/bhatiaabhinav/PPO.jl) and [Gym.jl](https://github.com/bhatiaabhinav/Gym.jl). The complete list of compatible packages is at the end of this README. The philosophy is that every package should add one small piece of functionality and work well with other packages. This is to keep the ecosystem lightweight and modular. An experiment that runs DQN on a CartPole should not include other heavy dependencies!

## Installation

```julia
using Pkg
Pkg.add("https://github.com/bhatiaabhinav/MDPs.jl")
```


## Examples

### Minimal Example

```julia
using MDPs
mdp = RandomDiscreteMDP(10, 2)  # 10 states, 2 actions
policy = RandomPolicy(mdp)
γ = 1.0
horizon = 100
max_trials = 10000
hooks = [ProgressMeterHook()]
episode_returns, episode_lengths = interact(mdp, policy, γ, horizon, max_trials, hooks...);
println("Average return: ", sum(episode_returns) / max_trials)
```

### An extended Example: Running ValueIteration and Q-Learning on a GridWorld

```julia
using MDPs
using GridWorlds: CliffWalkGridWorld
using ValueIteration: policy_evaluation, value_iteration
using TabularRL: QLearner
using StatsBase

mdp = CliffWalkGridWorld()  # From Sutton and Barto 2018. Example 6.6
display(mdp.grid)           # Show the gridworld
γ, horizon = 1.0, 50
random_baseline_empirical_value = interact(mdp, RandomPolicy(mdp), γ, horizon, 100000)[1] |> mean  # 100K episodes
random_baseline_exact_value = policy_evaluation(mdp, RandomPolicy(mdp), γ, horizon)[1]  # -104.388
println("Random baseline empirical value: ", random_baseline_empirical_value, ", exact value: ", random_baseline_exact_value)

# Value Iteration:
J_star, V_star, Q_star = value_iteration(mdp, γ, horizon)
optimal_policy = GreedyPolicy(Q_star)
optimal_policy_empirical_value = interact(mdp, optimal_policy, γ, horizon, 100000)[1] |> mean
println("Optimal policy empirical value: ", optimal_policy_empirical_value, ", exact value: ", J_star)  # -13.0
display(reshape(V_star, 4, 12))  # Show V* (reshaped to match the gridworld)

# Q-Learning:
nstates, nactions = length(state_space(mdp)), length(action_space(mdp)) # 48, 4
Q_star_estimates = zeros(nactions, nstates)
exploration_policy = EpsilonGreedyPolicy(Q_star_estimates, 0.1)  # Explore with probability 0.1
greedy_policy = GreedyPolicy(Q_star_estimates)
Q_learning_hook = QLearner(greedy_policy, Q_star_estimates, 0.99, 0.5)  # Learn Q* with discount factor 0.99 and learning rate 0.5
empirical_policy_evaluation_hook = EmpiricalPolicyEvaluationHook(greedy_policy, γ, horizon, 100, 1000)  # Empirically evaluates the greedy policy every 100 episodes (with sample size = 1000 episodes).
returns = interact(mdp, exploration_policy, γ, horizon, 1000, Q_learning_hook, empirical_policy_evaluation_hook, ProgressMeterHook())[1]

final_policy_empirical_value = interact(mdp, greedy_policy, γ, horizon, 100000)[1] |> mean
final_policy_exact_value = policy_evaluation(mdp, greedy_policy, γ, horizon)[1]
println("Final policy empirical value: ", final_policy_empirical_value, ", exact value: ", final_policy_exact_value)
V_star_estimates = maximum(Q_star_estimates, dims=1)  # V* estimates
display(reshape(V_star_estimates, 4, 12))  # Show V* estimates (reshaped to match the gridworld)
```

Plotting the learning curve is easy:
```julia
using Plots
plot(1:1000, returns, label="ep-greedy", xlabel="episodes", ylabel="return")
plot!(0:100:1000, empirical_policy_evaluation_hook.returns, label="greedy (mean)")
```

Recording Video:
```julia
interact(mdp, exploration_policy, γ, horizon, 4, VideoRecorderHook("videos/cliffwalk_ep_greedy", 1; format="gif"))  # record 4 episode of the epsilon-greedy policy
```


## List of compatible packages:

RL Environment Packages:

- [Bandits.jl](https://github.com/bhatiaabhinav/Bandits.jl): Provides multi-armed bandit environments.
- [GridWorlds.jl](https://github.com/bhatiaabhinav/GridWorlds.jl): Provides common gridworld environments and easy ways to create new ones.
- [ClassicControl.jl](https://github.com/bhatiaabhinav/ClassicControl.jl): Provides classic control environments like CartPole, MountainCar, Acrobot, etc.
- [Gym.jl](https://github.com/bhatiaabhinav/Gym.jl): Provides a wrapper for OpenAI Gym python environments.
- [MetaMDPs.jl](https://github.com/bhatiaabhinav/MetaMDPs.jl): Provides functionality to create a Meta MDP from a set of MDPs.
- [SemiCircle.jl](https://github.com/bhatiaabhinav/SemiCircle.jl): Provides the SemiCircle environment commonly used in meta reinforcement learning.


MDP Solvers and Reinforcement Learning Algorithms:
- [ValueIteration.jl](https://github.com/bhatiaabhinav/ValueIteration.jl): Provides value iteration and policy iteration algorithms.
- [TabularRL.jl](https://github.com/bhatiaabhinav/TabularRL.jl): Provides Q-Learning, SARSA, Expected SARSA, and other tabular RL algorithms.
- [DQN.jl](https://github.com/bhatiaabhinav/DQN.jl): Provides DQN and Double DQN algorithms, including with recurrent networks.
- [PPO.jl](https://github.com/bhatiaabhinav/PPO.jl): Provides A2C and PPO algorithm, including with recurrent and transformer networks.
- [SAC.jl](https://github.com/bhatiaabhinav/SAC.jl): Provides SAC and SAC-Discrete algorithm, including with recurrent networks.
