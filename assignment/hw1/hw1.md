## Assignment#1

### 1. Machine learning problem
**(a)**
1) BF
2) C
3) C
4) BG
5) AE
6) AD
7) BF
8) AE
9) BF

**(b)**
Answer: **This is definite False.** If you just choose parameters which performs best on whole dataset, it may cause the **overfitting problem or high variance**, that is say your model's generalization ability is low. So it's better to split the dataset into **training set、cross-validation set and test set**, we train our model parameters from training set that performs best on cross-validation set, meanwhile use the test set to examine the generalization ability of your model.

---

### 2. Bayes Decision Rule
**(a)**
- (i) Since we just have 3 boxes, so: $$P(B_1 = 1) = \frac{1}{3}$$
- (ii) Since we just have 1 box containes the bonus, so: $$P(B_2 = 0 \mid B_1 = 1) = 1$$
- (iii) From Bayes formula, we can get that: $$P(B_1 = 1 \mid B_2 = 0) = \frac{P(B_2 = 0 \mid B_1 = 1) \cdot P(B_1 = 1)}{P(B_2 = 0)} = \frac{1\ast(\frac{1}{3})}{(\frac{2}{3})} = \frac{1}{2}$$
- (iv) Since the $B_1$ and $B_3$ has no direct association, so that can get:$$P(B_1 = 1 \mid B_2 = 0) = P(B_3 = 1 \mid B_2 = 0) = \frac{1}{2}$$
That's to say, the probability of $B_1$ or $B_3$ that contains the bonus is just the same! So you can either choose stick to $B_1$ or choose the left box $B_3$, both are optimal choice.

**(b)**
- (i) The distribution of $P(\omega_i \mid x)$ is shown below:

![avatar](/Users/huangyifei/Desktop/data_mining/assignment/answer_images/hw1_2_b_i.png)
