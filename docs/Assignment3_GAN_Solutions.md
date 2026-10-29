# Assignment 3 GAN Theory

Xiaohan Zhao

## Question 1

For dilation 1, each output dimension is O = (I − 1)S − 2P + K + OP.

O = (8 − 1) × 2 − 2 × 1 + 4 + 1 = 14 − 2 + 4 + 1 = 17.

The output spatial size is 17 × 17.

## Question 2

Using the same input, kernel, padding, and output padding as Question 1:

O = (8 − 1) × 3 − 2 × 1 + 4 + 1 = 21 − 2 + 4 + 1 = 24.

The output changes from 17 × 17 to 24 × 24. In general, increasing stride from 2 to 3 adds I − 1 to each output dimension; here this adds 7.

## Question 3

With dilation 1, apply the formula separately to height and width:

Hout = (Hin − 1)Sh − 2Ph + Kh + OPh

Wout = (Win − 1)Sw − 2Pw + Kw + OPw

For identical settings in both dimensions: O = (I − 1)S − 2P + K + OP.

## Question 4

One possible configuration is a 2 × 2 kernel, stride 2, padding 0, and output padding 0, with dilation 1.

O = (16 − 1) × 2 − 2 × 0 + 2 + 0 = 30 + 2 = 32.

Therefore, a 16 × 16 feature map becomes 32 × 32.

## Question 5

Use the mini-batch population variance, dividing by 4 rather than 3.

Mean: μ = (6 + 8 + 10 + 6) / 4 = 7.5.

Deviations: [−1.5, 0.5, 2.5, −1.5].

Variance: σ² = (2.25 + 0.25 + 6.25 + 2.25) / 4 = 11 / 4 = 2.75.

Standard deviation: σ = √2.75 ≈ 1.6583.

Normalized output: (x − μ) / σ = [−3, 1, 5, −3] / √11

≈ [−0.9045, 0.3015, 1.5076, −0.9045].

The mean is 0 and variance is (9 + 1 + 25 + 9) / (4 × 11) = 1. Here ε is ignored for exact unit variance; implementations usually divide by √(σ² + ε) for numerical stability.

## Question 6

ReLU(x) = max(0, x): it equals x for x ≥ 0 and 0 for x < 0.

LeakyReLU(x) = x for x ≥ 0 and αx for x < 0, where α is a small positive slope (0.2 in this assignment).

The key difference is the slope for negative inputs: 0 for ReLU and α for LeakyReLU.

## Question 7

A ReLU neuron receives zero gradient whenever its input is negative. If it stays negative across the training data, it can stop learning, the “dying ReLU” problem. LeakyReLU retains a nonzero slope on the negative side, allowing gradients to pass and helping those neurons continue learning. This can improve gradient flow in deep networks, although it does not guarantee better performance.
