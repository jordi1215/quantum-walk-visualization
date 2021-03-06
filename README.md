# quantum-walk-visualization

This repo is used to compete in the 2021 Global Qiskit Hackathon Challenge

**Team Members:** Trisha Agrawal, James Larsen, Ronak Ramachandran, Jordi Ramos, Erika Tan

**Presentation Slides link:** [`Link to Google slides presentation`](https://docs.google.com/presentation/d/1Q2-ji42m3uzqoAu9EWHKb0BxtugebWO3WkQLOdyPyGg/edit?usp=sharing)

## Our Project

Our project aims to build a visualization tool for quantum walks on graphs. We first needed to build the mathematical tools to evolve our quantum walk. We then used the [`retworkx`](https://github.com/Qiskit/retworkx) from Qiskit to visualize our results. We produced it into a gif. format file that captures the snapshot per time step.

Our project includes a self-explanatory Python notebook which can be run in Jupyter, Google Colab, etc. Our code cells are supplemented by text explanations and diagrams to help the readers learn and understand the concept of quantum walk. Please refer to [`quantum_walk.ipynb`](quantum_walk.ipynb) for more information.

## Example

![](quantum_walk_demo.gif)

This is an example of a generated gif for a path on three vertices.

## Usage Guide

To import our visualization tool, download [`quantum_walk.py`](quantum_walk.py). Once this file is in your working directory, you can use the corresponding functions.

For example, you can run:

`import quantum_walk as qw`

to get the python file and run:

`qw.visualize_walk(adjacency_matrix, period, num_timesteps, "Name of Gif", "Snapshots Directory")`

with the appropriate variables to generate a gif of your quantum walk.

If you want a detailed walkthrough and other functionality, please refer to our python notebook file: [`quantum_walk.ipynb`](quantum_walk.ipynb) for more information.
