title: Performant Multi-Platform Reinforcement Learning
author:
  name: Nathan Epstein
  twitter: epstein_n
  url: http://nepste.in
  email: _@nepste.in

--
# Performant Multi-Platform Reinforcement Learning 

--
### The Project

Create a reinforcement learning library that is:

1) Simple to use

2) Fast

3) Multi-platform

https://github.com/NathanEpstein/Pavlov.js

--

### Two Distinct Pieces 

1) Reinforcement learning implementation

2) Tools for cross-language transpilation

--

# Reinforcement Learning

--

### Reinforcement Learning

Reinforcement learning is a type of machine learning in which software agents are trained to take actions in a given environment to maximize a cumulative reward.

--

### Markov Decision Process - Components

A Markov Decision process is a mathematical formalism that we will use to implement reinforcement learning. The relevant components of this formalism are
the __state space__, __action space__, __transition probabilities__, and __rewards__.

--

### State Space

The exhaustive set of possible states that a process can be in. Generally known a priori.

--

### Action Space

The exhaustive set of possible actions that can be taken (to influence the likelihood of transition between states). Generally known a priori.

--

### Transition Probabilities

The probabilities of transitioning between the various states given actions taken (specifically, a tensor, P, such that P_ijk = probabilitiy of going from state i to state k given action j). Generally not known a priori.

--

### Rewards

The rewards associated with occupying each state. Generally not known a priori.

--

### Markov Decision Process - Objectives

We are interested in understanding these elements in order to develop a __policy__; the set of actions we will take in each state.

Our goal is to determine a policy which produces the greatest possible cumulative rewards.

--

### Reward Parser

```cpp

#include "RewardParser.h"

std::vector<double> RewardParser::rewards() const {
  std::vector<double> total_state_rewards(d_state_count);
  std::vector<int> total_state_visits(d_state_count);

  const_obs_iter obs_it = d_obs -> begin();
  while (obs_it != d_obs -> end()) {
    int visits = obs_it -> state_transitions.size();
    double reward_per_visit = (obs_it -> reward) / visits;

    const_trans_iter trans_it = obs_it -> state_transitions.begin();
    while (trans_it != obs_it -> state_transitions.end()) {
      int state = trans_it -> encoded_state;
      total_state_rewards[state] += reward_per_visit;
      total_state_visits[state] += 1;

      ++trans_it;
    }

    ++obs_it;
  }

  std::vector<double> average_state_rewards;
  for (int i = 0; i < d_state_count; ++i) {
    double state_reward = total_state_rewards[i];
    if(total_state_visits[i] > 0) state_reward /= total_state_visits[i];

    average_state_rewards.push_back(state_reward);
  }

  return average_state_rewards;
}

```

--

### Transition Parser (part 1)

```cpp
#include "TransitionParser.h"

TransitionParser::TransitionParser(
  std::vector<observation> *observations,
  int state_count,
  int action_count)
  : d_obs(observations),
    d_state_count(state_count),
    d_action_count(action_count) {}

tensor TransitionParser::transition_probabilities() const {
  tensor transition_count = count_transitions();
  return parse_probabilities(transition_count);
}

```

--

### Transition Parser (part 2)

```cpp

tensor TransitionParser::count_transitions() const {

  tensor transition_count = makeTensor(d_state_count, d_action_count);

  obs_iter obs_it = d_obs -> begin();
  while (obs_it != d_obs -> end()) {
    trans_iter trans_it = obs_it -> state_transitions.begin();
    while (trans_it != obs_it -> state_transitions.end()) {
      int state = trans_it -> encoded_state;
      int action = trans_it -> encoded_action;
      int state_ = trans_it -> encoded_state_;

      transition_count[state][action][state_] += 1;

      ++trans_it;
    }
    ++obs_it;
  }

  return transition_count;
}
```

--

### Transition Parser (part 3)

```cpp
tensor TransitionParser::parse_probabilities(tensor &transition_count) const {

  tensor P = makeTensor(d_state_count, d_action_count);

  for (int state = 0; state < d_state_count; ++state) {
    for (int action = 0; action < d_action_count; ++action) {

      // count total_transitions
      int total_transitions = 0;
      for (int state_ = 0; state_ < d_state_count; ++state_)
      {
        total_transitions += transition_count[state][action][state_];
      }

      // parse parse probabilities from transitions
      for (int state_ = 0; state_ < d_state_count; ++state_)
      {
        if (total_transitions > 0) {
          double transitions = transition_count[state][action][state_];
          P[state][action][state_] = transitions / total_transitions;
        }
        else {
          P[state][action][state_] = 1.0 / d_state_count;
        }
      }
    }
  }

  return P;
}

```

--

### Policy Parser

```cpp

#include "PolicyParser.h"

std::vector<int> PolicyParser::policy(
  const tensor &P,
  const std::vector<double> &rewards,
  const double gamma = 0.9,
  const int iterations = 125) const
{
  std::vector<int> best_policy(d_state_count);
  std::vector<double> state_values(d_state_count);

  for (int i = 0; i < iterations; ++i) {
    for (int state = 0; state < d_state_count; ++state) {
      double state_value = -std::numeric_limits<double>::infinity();

      for (int action = 0; action < d_action_count; ++action) {
        double action_value = 0;

        for (int state_ = 0; state_ < d_state_count; ++state_) {
          action_value += (P[state][action][state_] * state_values[state_] * gamma);
        }

        if (action_value >= state_value) {
          state_value = action_value;
          best_policy[state] = action;
        }
      }
      state_values[state] = rewards[state] + state_value;
    }
  }

  return best_policy;
}

```

-- 

### Example: Array Search

- Given an array of sorted numbers, we would like to find a target value as quickly as possible.

- Our random numbers will be distributed exponentially. Can we use this information to do better than binary search?

--

### Approach

- Create many example searches (random, linear, and binary).
- For each step in the searches, create an observation.
- Input will be a tuple of current index, known floor, known ceiling, current value / target value (i.e. `{ 'location': 10, 'floor': 5, 'ceil': 12, 'ratio': 1.5 }`) encoded as a string.
- Output will be the desired step size to locate the target value.

--

### Results
<img src="./img/results.png">

random: 98.34, linear: 31.5, binary: 5.87, AI: 3.32

--


# Multi-platform code

-- 

### Why Transpile?

- Compiled C++ code is a highly performant choice for platforms that support it. 

- We can also target web and mobile by compiling to JavaScript (or Web Assembly).

- Additional platforms without additional code.

-- 

### JS Performance

In some cases, we can use a compiler to generate more performant JavaScript then what we are able to produce with "hand written" JS.

--

### Some things that make JS "slow"

- layout of data in physical memory
- ambiguous functions
- garbage collection
- unoptimized code

--

### Memory Layout

- In a typed language, elements of an array / vector will typically be stored in contiguous memory.

- This is unlikely to happen in JS because array elements have different types / sizes.

--

### Ambiguous Functions

- Functions with different argument types may have similar syntax but do very different things in physical memory (i.e. `let add = (a, b) => a + b;`).

--

### Garbage Collection

- Heap memory is not manually managed; may be reclaimed at unpredictable / undesirable times.

--

### Unoptimized Code
- "Hand written" code may contain room for optimizations that a compiler can perform (i.e. unnecessary computations and intermediate values, result reuse, memory locality, etc).

--

### Emscripten

- Allows us to compile LLVM-to-JS (C, C++, Rust).

- Can target WebAssembly or asm.js (optimizable low-level subset of JS).

- asm.js may be AOT or JIT compiled depending on the runtime.

--

### What Do We Get?

- Pre-emptive checks for type and linking errors.

- Libraries and syntax associated with the original language.

- **Sophisticated performance optimization that goes into the compiler.**

-- 

### Example (Pavlov Interface)


```cpp
#include "Pavlov.h"
#include <emscripten/bind.h>
using namespace emscripten;

// Implementation of Pavlov public interface...

EMSCRIPTEN_BINDINGS(pavlov) {
  class_<Pavlov>("Pavlov")
    .constructor<>()
    .function("transition", &Pavlov::transition)
    .function("reward", &Pavlov::reward)
    .function("learn", &Pavlov::learn)
    .function("action", &Pavlov::action)
    ;

  value_object<state_transition>("state_transition")
    .field("state", &state_transition::state)
    .field("state_", &state_transition::state_)
    .field("action", &state_transition::action)
    ;
}
```

https://github.com/NathanEpstein/Pavlov.js


