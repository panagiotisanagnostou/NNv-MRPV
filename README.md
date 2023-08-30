Neural Network voting Multiple Random Projections Voting (NNv-MRPV)
===================================================================

This repository contains the code and data related to the paper titled "Neural Network voting Multiple Random Projections Voting." The paper explores the use of random projections to enhance the performance of neural networks when dealing with single-cell RNA sequencing (scRNA-seq) data. This Markdown file serves as a guide to the repository structure and provides placeholders for additional information that will be added.

Usage
-----

For the execution of the source code, you simply need to execute the `executor.py` file. There are no needed parameters for the execution of the file. It automatically executes the NNv-MRPV algorithms and saves the results in the `res` folder. The results are saved in `.pkl` files, and the name of the file is the name of the dataset it comes from. The `.pkl` files are compressed pickle-dump files. Each file is a dictionary with the keys the name of the execution algorithm, and the values are dictionaries. The subsequent dictionaries have keys the prediction labels, the actual labels, the execution time of the model, the indexes of the test set and values of the models' scores for each of the 50 iterations of the experiment.


Citation
--------

If you use this paper, code, or data in your work, please cite:

TODO: Add the citation information for the paper once available.

Acknowledgment
--------------

Financed by the European Union - NextGenerationEU through Recovery and Resilience Facility, Greece 2.0, under the call RESEARCH – CREATE – INNOVATE (project code:TAEDK-06185 / T2EDK- 02800)

---

For any inquiries or issues related to this repository, please contact [Panagiotis Anagnostou](mailto:panagno@uth.gr), or create an issue in the repository.
