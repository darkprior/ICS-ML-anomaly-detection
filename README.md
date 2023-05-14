# ICS-ML-anomaly-detection
 Repository for the detection of cyber threats in ICS using machine learning. We used two machine learning algorithms:
 * Isolated forest
 * CNN

 We are dealing with an anomaly detection problem in ICS (PLC) network communication. Our goal is to detect changes in network traffic, intrusions and harmful code injection. We use network packets to train our models.

 The repository is in a prototype stage so expect frequent commits..

## Getting Started

### Prerequisites

List of Python libraries required to run the scripts:
* matplotlib
* numpy
* tensorflow
* sklearn
* pandas
* csv
* pickle



## Usage

The [Extraction script](./scripts/plcextracttest.py) is used to extract the packet files and store them in a binary numerical array. 
The [CNN classify script](./scripts/plccnntest.py) contains the CNN model as well as training / testing procedures.
The [Isolated forest script](./scripts/plcclassify.py) contains the isolated forest model with training / testing procedures.

<!-- ## Contributing

Guidelines on how to contribute to the project.

-->
## Authors

List of authors who have contributed to the project:

* [Denis Benka](https://www.linkedin.com/in/denis-benka/)
* [Sabína Vašová](https://www.linkedin.com/in/sabina-vasova/)

## License

This project is licensed under the [MIT License] license - see the [LICENSE.md](LICENSE.md) file for details.
