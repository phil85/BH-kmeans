# BH-kmeans

A k-means algorithm for clustering with soft must-link and cannot-link constraints. 

## Dependencies

BH-kmeans depends on:
* [Gurobi](https://anaconda.org/Gurobi/gurobi)
* [Numpy](https://anaconda.org/conda-forge/numpy)
* [Scipy](https://anaconda.org/anaconda/scipy)
* [Scikit-learn](https://anaconda.org/anaconda/scikit-learn)

Gurobi is a commercial mathematical programming solver. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/). 

## Installation

1) Download and install Gurobi (https://www.gurobi.com/downloads/)
2) Clone this repository (git clone https://github.com/phil85/BH-kmeans.git)

## Usage

The main.py file contains code that applies the BH-kmeans algorithm on an illustrative example.

```python
labels = bh_kmeans(X, n_clusters=2, ml=ml, cl=cl, p=p)
```

## Documentation

The documentation of the module bh_kmeans can be found [here](https://phil85.github.io/BH-kmeans/bh_kmeans.html).

## Reference

Please cite the following paper if you use this algorithm.

**Baumann, P. and Hochbaum, D.S.** (2021): A k-means algorithm for clustering with soft must-link and cannot-link constraints. Proceedings of the 11th International Conference on Pattern Recognition Applications and Methods, to appear. 

Bibtex:
```
@inproceedings{baumann2021kmeans,
	author={Philipp Baumann and Dorit S. Hochbaum},
	booktitle={Proceedings of the 11th International Conference on Pattern Recognition Applications and Methods},
	title={A k-means algorithm for clustering with soft must-link and cannot-link constraints},
	year={2021},
	notes={to appear},
}
```

## New constraint sets

In the above paper, we use new constraint sets for the data sets Iris and Wine. These constraint sets are provided in the folder new constraint sets.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


