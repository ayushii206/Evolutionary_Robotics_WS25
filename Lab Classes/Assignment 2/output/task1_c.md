### 📘 Mathematical Estimation (c)

Let:

* ( L = 33 ) — string length
* ( A = 27 ) — alphabet size (26 letters + space)

---

#### Step 1. Define the probability of improvement

At any point, assume ( k ) characters are correct.
To improve fitness, we must:

1. Pick one of the ( (L - k) ) incorrect characters.
2. Mutate it to the correct letter.

Hence, the probability of improvement is:

$$
p_{\text{improve}} = \frac{(L - k)}{L} \cdot \frac{1}{A}
$$

---

#### Step 2. Expected generations to improve fitness by 1

$$
E_k = \frac{1}{p_{\text{improve}}} = \frac{L \cdot A}{L - k}
$$

---

#### Step 3. Total expected generations

$$
E_{\text{total}} = \sum_{k=0}^{L-1} E_k = L \cdot A \sum_{k=0}^{L-1} \frac{1}{L - k}
= L \cdot A \cdot H_L
$$

where $$( H_L )$$ is the **harmonic number**, defined as:

$$
(H_L) {\approx \ln(L) + \gamma}
$$

and $$( \gamma \approx 0.577 )$$ (Euler–Mascheroni constant).

---

#### Step 4. Substitute the values

$$
E_{\text{total}} \approx 33 \times 27 \times (\ln(33) + 0.577)
$$

$$
E_{\text{total}} \approx 33 \times 27 \times 4.07 \approx 3620
$$

✅ **Expected average generations:** ≈ **3,620**

---

#### Step 5. Comparison with empirical results

| Source                 | Average Generations |
| ---------------------- | ------------------- |
| Analytical Estimate    | ≈ **3,620**         |
| Empirical (Simulation) | ≈ **6,300**         |

---

#### Step 6. Explanation of Difference

The difference arises because:

* Fitness improvements may **stall due to neutral moves** (mutations that do not change fitness).
* **Randomness** introduces variance in convergence time.
* Our model assumes **independent improvements** and **perfect acceptance** of all neutral moves.

Despite these simplifications, the **order of magnitude** (same scale of thousands) shows that the analytical estimation is **reasonable and consistent** with empirical results.

---
