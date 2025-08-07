# Hopfield Network with Sleep using PyTorch

This project implements a Hopfield network with sleep phases using the Hebbian learning rule. The implementation is done using Python.

## Table of Contents
- [Introduction](#introduction)
- [TODO](#todo)

## Introduction
Hopfield networks are a type of recurrent artificial neural network that can serve as content-addressable memory systems. This project explores the concept of incorporating sleep phases into the Hopfield network to enhance its performance and stability.

## TODO

- Find the performance bottlenecks and optimize if possible/easy, using `cProfile`. Evaluate vectorizations/parallelizations/numba llvm optimizations/etc... 

- Refactor repo

- Hopfield theory summary (Definition of $\alpha$, what constitutes a correctly retrieved pattern? minimal basin size, understand phases, unlearning, observables, spurious states, spin glass theory derivation of sleep)

- Read paper: how do correlated datasets affect learning?

- Structure the project presentation (LateX/Manim)

- Prepare the final results/visualizations

### Presentation
- Motivation (why hopfield, why sleep? Why train with stimuli instead of Hebbian weights? Parallels with real associative memory with images)

- Theory of sleep with spin glasses

- Diff equation of sleep

- Simulation results (capacity->consolidation, spurious states -> unlearning. Correlated data. One animation of a cifar10 retrieval)

### Advanced
- Very briefly study spin glass calculations for sleep

- Study the convergence of the hebbian wake phase analytically

- What could be animated about the project presentation?

- Scaling of hyperparameters

- Classification with Hopfield

- Cifar10 performance