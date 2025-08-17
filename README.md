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
```├── 1
├── : Python Code for Binomial Probability (using scipy)
├── A bag has 5 red and 3 green balls. What’s the probability of picking 2 red balls without replacement?
├── Bernoulli Distribution Simulation
├── Binomial Distribution (Probability of 3 successes in 5 trials)
├── Normal Distribution – Z-Score Calculation
├── Probability & Statistics Examples in Python
├── Probability Simulation (Coin Toss)
├── Probability Tree Calculation python Copy Edit
├── Python Code for CDF and PDF of Normal Distribution
├── README.md
├── basic prob
├── bayes theoram
├── bayes theorm1
├── bernoulli example
├── bino
├── binom use
├── byess for disease
├── cdf example
├── clt
├── coin_toss
├── discrete uniform distribution example
├── from scipy.stats import binom
├── hypergeom
├── nbinom
├── pdf example
├── pmf example
├── poission distribution
├── poisson distribution exxample
├── probability-solved
    ├── README.md
    ├── bayes_theorem.py
    └── conditional_probability.py
└── rollDiceProbability


/1:
--------------------------------------------------------------------------------
1 | from scipy.stats import binom
2 | 
3 | n = 3
4 | k = 2
5 | p = 0.7
6 | 
7 | probability = binom.pmf(k, n, p)
8 | print(probability)
9 | 


--------------------------------------------------------------------------------
/: Python Code for Binomial Probability (using scipy):
--------------------------------------------------------------------------------
 1 | from scipy.stats import binom
 2 | 
 3 | # Probability of exactly 3 successes in 10 trials with p = 0.4
 4 | n = 10
 5 | p = 0.4
 6 | x = 3
 7 | 
 8 | prob = binom.pmf(x, n, p)
 9 | print(f"Probability of 3 successes: {prob}")
10 | 


--------------------------------------------------------------------------------
/A bag has 5 red and 3 green balls. What’s the probability of picking 2 red balls without replacement?:
--------------------------------------------------------------------------------
 1 | from math import comb
 2 | 
 3 | # Total ways to choose 2 balls from 8
 4 | total_ways = comb(8, 2)
 5 | 
 6 | # Ways to choose 2 red balls from 5
 7 | favorable_ways = comb(5, 2)
 8 | 
 9 | prob = favorable_ways / total_ways
10 | print("Probability of choosing 2 red balls:", prob)
11 | 


--------------------------------------------------------------------------------
/Bernoulli Distribution Simulation:
--------------------------------------------------------------------------------
 1 | from scipy.stats import bernoulli
 2 | import matplotlib.pyplot as plt
 3 | 
 4 | p = 0.7  # probability of success
 5 | data = bernoulli.rvs(p, size=1000)
 6 | 
 7 | plt.hist(data, bins=2, edgecolor='black')
 8 | plt.title("Bernoulli Distribution (p=0.7)")
 9 | plt.xticks([0, 1], ['Failure (0)', 'Success (1)'])
10 | plt.xlabel("Outcome")
11 | plt.ylabel("Frequency")
12 | plt.show()
13 | 


--------------------------------------------------------------------------------
/Binomial Distribution (Probability of 3 successes in 5 trials):
--------------------------------------------------------------------------------
1 | from scipy.stats import binom
2 | 
3 | n = 5
4 | p = 0.6
5 | k = 3
6 | 
7 | prob = binom.pmf(k, n, p)
8 | print("P(X = 3):", prob)
9 | 


--------------------------------------------------------------------------------
/Normal Distribution – Z-Score Calculation:
--------------------------------------------------------------------------------
 1 | import scipy.stats as stats
 2 | 
 3 | x = 75
 4 | mean = 70
 5 | std = 10
 6 | 
 7 | z = (x - mean) / std
 8 | prob = stats.norm.cdf(z)
 9 | print("P(X ≤ 75):", prob)
10 | 


--------------------------------------------------------------------------------
/Probability & Statistics Examples in Python:
--------------------------------------------------------------------------------
 1 | """
 2 | Probability & Statistics Examples in Python
 3 | Author: Aadarsh Tiwari
 4 | Description: Collection of 10 examples demonstrating
 5 |              probability and statistics concepts in Python.
 6 | """
 7 | 
 8 | # 1. Probability of Specific Event (Basic Rule)
 9 | favorable_outcomes = 1
10 | total_outcomes = 6
11 | probability = favorable_outcomes / total_outcomes
12 | print("1. P(rolling a 4):", probability)
13 | 
14 | # 2. Complement Rule
15 | P_heads = 0.5
16 | P_not_heads = 1 - P_heads
17 | print("2. P(Not Heads):", P_not_heads)
18 | 
19 | # 3. Addition Rule for Mutually Exclusive Events
20 | P_A = 0.3
21 | P_B = 0.4
22 | P_A_or_B = P_A + P_B
23 | print("3. P(A or B) [Mutually Exclusive]:", P_A_or_B)
24 | 
25 | # 4. Addition Rule for Non-Mutually Exclusive Events
26 | P_A = 0.5
27 | P_B = 0.6
28 | P_A_and_B = 0.3
29 | P_A_or_B = P_A + P_B - P_A_and_B
30 | print("4. P(A or B) [Non-Mutually Exclusive]:", P_A_or_B)
31 | 
32 | # 5. Multiplication Rule for Independent Events
33 | P_A = 0.4
34 | P_B = 0.5
35 | P_A_and_B = P_A * P_B
36 | print("5. P(A and B) [Independent]:", P_A_and_B)
37 | 
38 | # 6. Multiplication Rule for Dependent Events
39 | P_A = 0.5
40 | P_B_given_A = 0.3
41 | P_A_and_B = P_A * P_B_given_A
42 | print("6. P(A and B) [Dependent]:", P_A_and_B)
43 | 
44 | # 7. Conditional Probability
45 | P_A_and_B = 0.2
46 | P_B = 0.4
47 | P_A_given_B = P_A_and_B / P_B
48 | print("7. P(A|B):", P_A_given_B)
49 | 
50 | # 8. Bayes’ Theorem
51 | P_D = 0.02          # P(Disease)
52 | P_not_D = 0.98      # P(No Disease)
53 | P_Pos_given_D = 0.90        # P(Positive | Disease)
54 | P_Pos_given_not_D = 0.05    # P(Positive | No Disease)
55 | 
56 | numerator = P_Pos_given_D * P_D
57 | denominator = (P_Pos_given_D * P_D) + (P_Pos_given_not_D * P_not_D)
58 | P_D_given_Pos = numerator / denominator
59 | print("8. P(Disease | Positive):", round(P_D_given_Pos, 4))
60 | 
61 | # 9. Binomial Probability
62 | from scipy.stats import binom
63 | n = 15
64 | p = 0.3
65 | k = 5
66 | probability = binom.pmf(k, n, p)
67 | print(f"9. P(X = {k} successes in {n} trials):", probability)
68 | 
69 | # 10. Normal Distribution Probability
70 | import scipy.stats as stats
71 | mean = 50
72 | std_dev = 10
73 | x = 60
74 | z_score = (x - mean) / std_dev
75 | P_less_than_x = stats.norm.cdf(z_score)
76 | print(f"10. P(X ≤ {x}):", P_less_than_x)
77 | 


--------------------------------------------------------------------------------
/Probability Simulation (Coin Toss):
--------------------------------------------------------------------------------
1 | import random
2 | 
3 | def coin_toss_simulation(trials=1000):
4 |     heads = sum(1 for _ in range(trials) if random.choice(['H', 'T']) == 'H')
5 |     probability = heads / trials
6 |     return probability
7 | 
8 | print("Estimated Probability of Heads:", coin_toss_simulation(100000))
9 | 


--------------------------------------------------------------------------------
/Probability Tree Calculation python Copy Edit:
--------------------------------------------------------------------------------
 1 | # Probability of events
 2 | P_A = 0.6  # Probability of rain
 3 | P_B_given_A = 0.8  # Probability of carrying umbrella if rain
 4 | P_B_given_not_A = 0.2  # Probability of carrying umbrella if no rain
 5 | 
 6 | # Total probability of carrying umbrella
 7 | P_B = P_A * P_B_given_A + (1 - P_A) * P_B_given_not_A
 8 | 
 9 | print("Probability of carrying umbrella:", P_B)
10 | 


--------------------------------------------------------------------------------
/Python Code for CDF and PDF of Normal Distribution:
--------------------------------------------------------------------------------
 1 | from scipy.stats import norm
 2 | 
 3 | mean = 0
 4 | std_dev = 1
 5 | 
 6 | # PDF at x = 1
 7 | pdf_val = norm.pdf(1, mean, std_dev)
 8 | 
 9 | # CDF at x = 1
10 | cdf_val = norm.cdf(1, mean, std_dev)
11 | 
12 | print("PDF at x=1:", pdf_val)
13 | print("CDF at x=1:", cdf_val)
14 | 




--------------------------------------------------------------------------------
/basic prob:
--------------------------------------------------------------------------------
1 | **Topic:** Basic Probability  
2 | **Question:** What is the probability of drawing an Ace from a standard deck?  
3 | **Answer:**  
4 | P(Ace) = Number of Aces / Total Cards = 4/52 = 1/13 ≈ 0.077
5 | 


--------------------------------------------------------------------------------
/bayes theoram:
--------------------------------------------------------------------------------
 1 | """
 2 | 📘 Bayes' Theorem - Real World Example: Medical Testing
 3 | 
 4 | This script demonstrates Bayes' Theorem to calculate the probability
 5 | of actually having a disease given a positive test result.
 6 | 
 7 | Problem:
 8 | - A disease affects 2% of the population.
 9 | - The test has:
10 |     ✅ 90% True Positive Rate (Sensitivity)
11 |     ❌ 5% False Positive Rate (False alarm when person is healthy)
12 | 
13 | Goal:
14 | - Given a person tests positive, what is the probability they truly have the disease?
15 | """
16 | 
17 | # Given probabilities
18 | P_D = 0.02                      # P(Disease)
19 | P_not_D = 1 - P_D               # P(No Disease)
20 | P_Pos_given_D = 0.90           # P(Positive | Disease)
21 | P_Pos_given_not_D = 0.05       # P(Positive | No Disease)
22 | 
23 | # Bayes’ Theorem: P(Disease | Positive)
24 | numerator = P_Pos_given_D * P_D
25 | denominator = (P_Pos_given_D * P_D) + (P_Pos_given_not_D * P_not_D)
26 | P_D_given_Pos = numerator / denominator
27 | 
28 | # Print the result
29 | print("📊 Probability of having disease given a positive test result:")
30 | print("P(Disease | Positive) =", round(P_D_given_Pos, 4))  # Rounded to 4 decimal places
31 | 


--------------------------------------------------------------------------------
/bayes theorm1:
--------------------------------------------------------------------------------
 1 | # Given values
 2 | P_D = 0.02          # P(Disease)
 3 | P_not_D = 0.98      # P(No Disease)
 4 | P_Pos_given_D = 0.90        # P(Positive | Disease)
 5 | P_Pos_given_not_D = 0.05    # P(Positive | No Disease)
 6 | 
 7 | # Bayes Theorem
 8 | numerator = P_Pos_given_D * P_D
 9 | denominator = (P_Pos_given_D * P_D) + (P_Pos_given_not_D * P_not_D)
10 | P_D_given_Pos = numerator / denominator
11 | 
12 | print("P(Disease | Positive):", round(P_D_given_Pos, 4))
13 | 


--------------------------------------------------------------------------------
/bernoulli example:
--------------------------------------------------------------------------------
 1 | """
 2 | ✅ Bernoulli Distribution – Single Trial Probability
 3 | 
 4 | This script calculates success/failure probability using Bernoulli distribution.
 5 | """
 6 | 
 7 | from scipy.stats import bernoulli
 8 | 
 9 | # Probability of success (e.g., getting Heads)
10 | p = 0.7
11 | 
12 | # Create Bernoulli distribution
13 | dist = bernoulli(p)
14 | 
15 | # Probabilities
16 | success_prob = dist.pmf(1)  # P(X = 1)
17 | failure_prob = dist.pmf(0)  # P(X = 0)
18 | 
19 | print("✅ Bernoulli Distribution Example")
20 | print(f"P(Success) = {success_prob}")
21 | print(f"P(Failure) = {failure_prob}")
22 | 


--------------------------------------------------------------------------------
/bino:
--------------------------------------------------------------------------------
1 | from scipy.stats import binom
2 | 
3 | n = 10
4 | p = 0.4
5 | k = 6
6 | 
7 | probability = binom.pmf(k, n, p)
8 | print(probability)
9 | 


--------------------------------------------------------------------------------
/binom use:
--------------------------------------------------------------------------------
1 | from scipy.stats import binom
2 | 
3 | n = 15     # Total number of trials
4 | p = 0.3    # Probability of success in each trial
5 | k = 5      # Number of desired successes
6 | 
7 | probability = binom.pmf(k, n, p)
8 | print(probability)
9 | 


--------------------------------------------------------------------------------
/byess for disease:
--------------------------------------------------------------------------------
 1 | # Given values
 2 | P_D = 0.02          # P(Disease)
 3 | P_not_D = 0.98      # P(No Disease)
 4 | P_Pos_given_D = 0.90        # P(Positive | Disease)
 5 | P_Pos_given_not_D = 0.05    # P(Positive | No Disease)
 6 | 
 7 | # Bayes' Theorem
 8 | numerator = P_Pos_given_D * P_D
 9 | denominator = (P_Pos_given_D * P_D) + (P_Pos_given_not_D * P_not_D)
10 | P_D_given_Pos = numerator / denominator
11 | 
12 | print("P(Disease | Positive):", round(P_D_given_Pos, 4))
13 | 


--------------------------------------------------------------------------------
/cdf example:
--------------------------------------------------------------------------------
 1 | """
 2 | 📉 CDF – Cumulative Distribution Function
 3 | 
 4 | This script demonstrates how to calculate CDF using Normal distribution.
 5 | """
 6 | 
 7 | from scipy.stats import norm
 8 | 
 9 | # Example:
10 | # Height of students is normally distributed with mean 170 cm and std 10 cm.
11 | # What is the probability a student is shorter than or equal to 180 cm?
12 | 
13 | mu = 170     # mean
14 | sigma = 10   # standard deviation
15 | x = 180      # value for CDF
16 | 
17 | # CDF = P(X <= x)
18 | cdf_value = norm.cdf(x, mu, sigma)
19 | 
20 | print("📉 CDF Example (Normal Distribution)")
21 | print(f"P(X <= {x}) in N({mu}, {sigma}²) =", round(cdf_value, 4))
22 | 


--------------------------------------------------------------------------------
/clt:
--------------------------------------------------------------------------------
 1 | import numpy as np
 2 | import matplotlib.pyplot as plt
 3 | 
 4 | population = np.random.exponential(scale=2, size=10000)
 5 | sample_means = [np.mean(np.random.choice(population, size=30)) for _ in range(1000)]
 6 | 
 7 | plt.hist(sample_means, bins=30, edgecolor='black')
 8 | plt.title("Central Limit Theorem – Sampling Distribution")
 9 | plt.xlabel("Sample Means")
10 | plt.ylabel("Frequency")
11 | plt.show()
12 | 


--------------------------------------------------------------------------------
/coin_toss:
--------------------------------------------------------------------------------
1 | import random
2 | 
3 | # Simulate 100 coin tosses
4 | tosses = [random.choice(['Heads', 'Tails']) for _ in range(100)]
5 | prob_heads = tosses.count('Heads') / 100
6 | 
7 | print("Experimental Probability of Heads:", prob_heads)
8 | 


--------------------------------------------------------------------------------
/discrete uniform distribution example:
--------------------------------------------------------------------------------
 1 | """
 2 | ✅ Discrete Uniform Distribution – Equal Likelihood Example
 3 | 
 4 | This script calculates the probability of getting a specific value from a fair die.
 5 | """
 6 | 
 7 | from scipy.stats import randint
 8 | 
 9 | # Define uniform distribution for a fair die: values 1 to 6
10 | low = 1
11 | high = 7  # upper bound is exclusive in scipy's randint
12 | 
13 | die_distribution = randint(low, high)
14 | 
15 | # Get probability of rolling a 3
16 | value = 3
17 | prob = die_distribution.pmf(value)
18 | 
19 | print("🎲 Uniform Distribution Example")
20 | print(f"Probability of rolling a {value} = {prob}")
21 | 


--------------------------------------------------------------------------------
/from scipy.stats import binom:
--------------------------------------------------------------------------------
 1 | from scipy.stats import binom
 2 | 
 3 | n = 5    # number of trials
 4 | p = 0.2  # probability of success
 5 | k = 2    # number of successes
 6 | 
 7 | # PMF - Probability Mass Function
 8 | probability = binom.pmf(k, n, p)
 9 | 
10 | print(probability)
11 | 


--------------------------------------------------------------------------------
/hypergeom:
--------------------------------------------------------------------------------
 1 | from scipy.stats import hypergeom
 2 | 
 3 | # Parameters
 4 | N = 10     # total items
 5 | K = 4      # total success items
 6 | n = 3      # number of draws
 7 | k = 2      # number of success in draws
 8 | 
 9 | # Hypergeometric Probability
10 | prob = hypergeom.pmf(k, N, K, n)
11 | print(f"Probability of getting exactly {k} red balls: {prob:.4f}")
12 | 


--------------------------------------------------------------------------------
/nbinom:
--------------------------------------------------------------------------------
 1 | from scipy.stats import nbinom
 2 | 
 3 | # Parameters
 4 | r = 3  # number of successes
 5 | p = 0.4  # probability of success
 6 | x = 5  # number of failures
 7 | 
 8 | # PMF
 9 | prob = nbinom.pmf(x, r, p)
10 | print(f"Probability of 5 failures before 3 successes: {prob:.4f}")
11 | 


--------------------------------------------------------------------------------
/pdf example:
--------------------------------------------------------------------------------
 1 | """
 2 | 📈 PDF – Probability Density Function (Continuous Random Variable)
 3 | 
 4 | This script shows how to calculate PDF using a Normal (Gaussian) distribution.
 5 | """
 6 | 
 7 | from scipy.stats import norm
 8 | 
 9 | # Example scenario:
10 | # A dataset of heights is normally distributed with mean 170 cm and std deviation 10 cm.
11 | # What is the density (PDF) at 180 cm?
12 | 
13 | mu = 170     # mean
14 | sigma = 10   # standard deviation
15 | x = 180      # point to calculate PDF at
16 | 
17 | # PDF = f(x)
18 | pdf_value = norm.pdf(x, mu, sigma)
19 | 
20 | print("📈 PDF Example (Normal Distribution)")
21 | print(f"PDF at x = {x} in N({mu}, {sigma}²) =", round(pdf_value, 4))
22 | 


--------------------------------------------------------------------------------
/pmf example:
--------------------------------------------------------------------------------
 1 | """
 2 | 🎯 PMF – Probability Mass Function (Discrete Random Variable)
 3 | 
 4 | This script shows how to calculate the PMF using the Binomial distribution.
 5 | """
 6 | 
 7 | from scipy.stats import binom
 8 | 
 9 | # Example scenario:
10 | # Toss a coin 10 times, what's the probability of getting exactly 4 heads?
11 | 
12 | n = 10         # total trials
13 | p = 0.5        # probability of success (getting a head)
14 | k = 4          # number of desired successes
15 | 
16 | # PMF = P(X = k)
17 | pmf_value = binom.pmf(k, n, p)
18 | 
19 | print("🎯 PMF Example")
20 | print(f"P(X = {k}) in Binomial({n}, {p}) =", round(pmf_value, 4))
21 | 


--------------------------------------------------------------------------------
/poission distribution:
--------------------------------------------------------------------------------
 1 | import matplotlib.pyplot as plt
 2 | import numpy as np
 3 | 
 4 | λ = 4
 5 | x = np.arange(0, 15)
 6 | pmf_values = poisson.pmf(x, λ)
 7 | 
 8 | plt.bar(x, pmf_values)
 9 | plt.title("Poisson Distribution (λ=4)")
10 | plt.xlabel("Number of events (x)")
11 | plt.ylabel("Probability")
12 | plt.grid(True)
13 | plt.show()
14 | 


--------------------------------------------------------------------------------
/poisson distribution exxample:
--------------------------------------------------------------------------------
 1 | """
 2 | 📈 Poisson Distribution – Rare Events Modeling
 3 | 
 4 | This script demonstrates how to calculate Poisson probability.
 5 | """
 6 | 
 7 | from scipy.stats import poisson
 8 | 
 9 | # Example:
10 | # A call center receives 4 calls per hour on average.
11 | # What’s the probability they get exactly 6 calls in an hour?
12 | 
13 | λ = 4  # average rate (lambda)
14 | k = 6  # number of actual events
15 | 
16 | # P(X = k)
17 | probability = poisson.pmf(k, λ)
18 | 
19 | print("📈 Poisson Distribution Example")
20 | print(f"P(X = {k}) when λ = {λ} =", round(probability, 4))
21 | 


--------------------------------------------------------------------------------
/probability-solved/README.md:
--------------------------------------------------------------------------------
https://raw.githubusercontent.com/aadarsh3419/probability/757ad738333c4d251c1b9bed572c2dcb1eab18be/probability-solved/README.md


--------------------------------------------------------------------------------
/probability-solved/bayes_theorem.py:
--------------------------------------------------------------------------------
https://raw.githubusercontent.com/aadarsh3419/probability/757ad738333c4d251c1b9bed572c2dcb1eab18be/probability-solved/bayes_theorem.py


--------------------------------------------------------------------------------
/probability-solved/conditional_probability.py:
--------------------------------------------------------------------------------
https://raw.githubusercontent.com/aadarsh3419/probability/757ad738333c4d251c1b9bed572c2dcb1eab18be/probability-solved/conditional_probability.py


--------------------------------------------------------------------------------
/rollDiceProbability:
--------------------------------------------------------------------------------
 1 | import random
 2 | 
 3 | class Solution(object):
 4 |     def rollDiceProbability(self, trials=10000):
 5 |         count = 0
 6 |         for _ in range(trials):
 7 |             if random.randint(1, 6) + random.randint(1, 6) == 7:
 8 |                 count += 1
 9 |         return count / trials
10 | 
11 | # Example
12 | sol = Solution()
13 | print(sol.rollDiceProbability(100000))  # Approx 0.167
14 | 


--------------------------------------------------------------------------------

## 🧪 Sample Code – Binomial Distribution (PMF)

```python
from scipy.stats import binom

n = 5    # Total trials
p = 0.2  # Probability of success
k = 2    # Exact successes

prob = binom.pmf(k, n, p)
print(f"Probability of getting exactly {k} successes out of {n} trials: {prob}")
