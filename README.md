# Clustering-For-Federated-Learning-With-ALOHA

This repository contains all the codes that was created to generate the results of the Master's Thesis document entitled "A distributed D2D clustering algorithm tailored for hierarchical federated learning in a multichannel ALOHA network".

The document can be found in: "https://repositorio.utfpr.edu.br/jspui/handle/1/34706"

Choi's reference article can be found in: "https://arxiv.org/abs/1912.06273"

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Algorithm Overview](#algorithm-overview)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Federated Learning (FL) allows multiple clients to collaboratively train a machine learning model without sharing raw data, preserving privacy. In a Hierarchical FL (HFL) system, clients are grouped into clusters, with each cluster having its own model aggregation before the global aggregation. Traditional clustering algorithms, such as K-Means or DBSCAN, does not take into consideration some important aspects (business rules) of real systems, or the proposed system model, therefore they are not efficient enough.
This way, this project presents a novel clustering algorithm that considers some important aspects of a scenario composed of HFL that uses a multichannel ALOHA protocol to improve the communication efficiency and overall performance of HFL systems.

## Project Structure

.
├── Clustering
│ ├── proposed_clustering_algorithm.py # Main python code implementing the creation of the devices and the clustering algorithm
│ └── proposed_clustering_algorithm_step_by_step.ipynb # Notebook for visualizing the results step-by-step of each proposed process of the algorithm
├── Models
│ └── models_arrangement.py # Python script with the 3 protocol models implemented by Choi, with and without the proposed D2D clustering algorithm
├── Runs # Folder for storing simulation results and analysis
├── LICENSE
├── README.md # Project documentation
├── main.ipynb # Main Jupyter notebook for running the simmulations
└── requirements.txt # List of dependencies and libraries required

## Installation

To run the folder, enter the source, where the main.ipynb file is, and run the same file. To run it, one could use google colab or run it locally.
Depending on what you choose, there is a small change in the importings, inside the main file, but there is a comment for that. Install all the libraries in the requirements:

"pip install -r requirements.txt"

## Algorithm Overview

The proposed clustering algorithm groups clients in the HFL system based on their communication capabilities and learning characteristics. By leveraging the multichannel Aloha protocol, the algorithm aims to reduce communication collisions and improve overall system efficiency.

Key Features:
- Dynamic Clustering: Adaptive clustering of clients based on real-time network conditions.
- Multichannel Aloha: Utilization of multichannel Aloha to manage communication between clients and the server.
- Scalability: The algorithm is designed to scale with the number of clients and channels in the system.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are encouraged.

Fork the project
Create a feature branch (git checkout -b feature/YourFeature)
Commit your changes (git commit -m 'Add YourFeature')
Push to the branch (git push origin feature/YourFeature)
Open a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
