# 🎲 Probability in Python – Aadarsh's Full Practice Zone

![Project Pipeline](https://media.licdn.com/dms/image/v2/C4D22AQEtvYIdr07T5g/feedshare-shrink_800/feedshare-shrink_800/0/1642464540245?e=1756944000&v=beta&t=5gpIkeBU2Ay2p9UXMdCnCz5t7f2cgNxjmX5lGxaOPkU))


Welcome to the **Probability** section of my Data Science journey.  
This folder contains **all the probability topics I'm learning** — from basics to advanced — implemented in Python with clear examples, real-world logic, formulas, and clean code.

Every concept is written in a beginner-friendly way so anyone can learn and understand.

---

## 📚 Topics Covered (Complete Roadmap)

- ✅ **Basic Probability Rules**
- ✅ **Tree Diagrams**
- ✅ **Permutation & Combination**
- ✅ **Probability Laws**
- ✅ **Bayes’ Theorem**
- ✅ **Naive Bayes Algorithm**
- ✅ **Bernoulli Distribution**
- ✅ **Binomial Distribution**
- ✅ **Geometric Distribution**
- ✅ **Negative Binomial Distribution**
- ✅ **Poisson Distribution**
- ✅ **Hypergeometric Distribution**
- ✅ **Probability Density Functions (PDF)**
- ✅ **Cumulative Distribution Functions (CDF)**
- ✅ **Real-world Applications & Python Simulations**

> 🛠 Each topic includes:
> - ✍️ Simple explanation  
> - 📘 Real-world examples  
> - 🧮 Core formulas  
> - 🐍 Python implementation using `scipy.stats`, `numpy`, etc.

---

## 🧪 Sample Code – Binomial Distribution (PMF)

```python
from scipy.stats import binom

n = 5    # Total trials
p = 0.2  # Probability of success
k = 2    # Exact successes

prob = binom.pmf(k, n, p)
print(f"Probability of getting exactly {k} successes out of {n} trials: {prob}")
