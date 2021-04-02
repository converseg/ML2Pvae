---
title: 'ML2Pvae: An R package for high-dimensional Item Response Theory parameter estimation'
tags:
  - R
  - educational measurement
  - item response theory
  - variational autoencoders
authors:
  - name: Geoffrey Converse^[Corresponding author]
    orcid: 0000-0001-8764-9950
    affiliation: 1
  - name: Mariana Curi
    orcid: 0000-0002-7651-1064
    affiliation: 2
  - name: Suely Oliveira
    orcid: 0000-0003-4987-5353
    affiliation: 1
affiliations:
  - name: University of Iowa
    index: 1
  - name: University of São Paulo
    index: 2
date: 28 March 2021
bibliography: paper.bib

---

# Summary

In educational measurement, Item Response Theory (IRT) aims to model the probability of a student answering a question (item) correctly as a function of the student's continuous latent ability (skill) values, along with parameters associated with the item itself such as the question difficulty. For example, the basic arithmetic question $3 \times 4 + 2$ requires the skills "multiplication" and "addition". A popular IRT model, and the focus of the software `ML2Pvae`, is the Multidimensional Logistic 2-Parameter (ML2P) model, which gives the probability of student $j$ answering item $i$ correctly as
\begin{equation}\label{eq:ml2p}
P(x_{ij} = 1 | \vec \theta_j; \vec a_i, b_i) = \frac{1}{1 + \exp\left(-\sum_{k=1}^K a_{ik} \theta_{jk} + b_i\right)}
\end{equation}
where $\vec \theta_j =\{\theta_{jk}\}_{k=1}^K$ quantifies student $j$'s $K$ latent abilities, $a_{ik} \in \vec a_i=\{a_{ik}\}_{k=1}^K$ determines how well item $i$ can discern between individuals with small differences in their level of ability $k$, and $b_i$ gives the difficulty of item $i$.

In large-scale assessments (e.g. standardized testing), it is a common goal for researchers to quantify the associated parameters, both to better measure each student's ability in various topics and determine the efficacy of each assessment item. Estimating the parameters $\vec \theta_j$, $\vec a_i$, and $b_i$ for all students $j$ and all items $i$ is a well-studied task, many of which involve some sort of Monte-Carlo method to evaluate integrals.

# Statement of need

Software like `mirt` [@mirt] is popular for estimating IRT parameters, but due to the curse of dimensionality, such methods experience difficulty or are infeasible on assessments where the dimension of the latent ability space is large, i.e. a large number ($>8$) of distinct latent abilities are under assessment. Recent work has explored the use of a modified variational autoencoder (VAE), a type of neural network, to estimate parameters [@Curi:2019]. This method, ML2P-VAE, does not require numerical integration and thus does not have trouble with high-dimensional data. Further extensions generalize ML2P-VAE method to correlated latent abilities [@Converse:2021] and modify the method to fit an exploratory setting [@Urban:2020].

The `ML2Pvae` package for R [@R] provides IRT practitioners with easy-to-use functions construct, train, and evaluate VAE models for parameter estimation without requiring background knowledge of neural networks or TensorFlow [@tensorflow]. Additionally, `ML2Pvae` also includes the capability to fit a VAE to a more general multivariate Gaussian latent distribution with non-identity covariance matrix. This novel neural network architecture implementation could be useful to researchers in other applications outside of educational measurement.

Source code and code examples for the R package `ML2Pvae` [@ml2pvae_software] can be found at [https://github.com/converseg/ML2Pvae](https://github.com/converseg/ML2Pvae). `ML2Pvae` is also available on CRAN and can be installed directly in R via the command `install.packages('ML2Pvae')`.



# References

