This repository includes data and code for the KDEm algorithm in following paper:

Mengting Wan, Xiangyu Chen, Lance Kaplan, Jiawei Han, Jing Gao, Bo Zhao, "From Truth Discovery to Trustworthy Opinion Discovery: An Uncertainty-Aware Quantitative Modeling Approach", in Proc. of 2016 ACM SIGKDD Conf. on Knowledge Discovery and Data Mining (KDD'16), San Francisco, CA, Aug. 2016

If you have any questions, feel free to contact me at m5wan@ucsd.edu

To run experiments on synthetic datasets -- Synthetic(unimodal) and Synthetic(mix), you can directly 

         type in "python test.py -synuni" or "python test.py -synmix"

Notice that experiments may run for a while.

To run experiments on real-world datasets -- Population(outlier) and Tripadvisor, you must download the Population(outlier) data from this [link](http://cogcomp.cs.illinois.edu/page/resource_view/16) and Tripadvisor data from this [link](http://times.cs.uiuc.edu/~wang296/Data/).

For Population(outlier), please put two files "popAnswerOut.txt" and "pupTuples.txt" in the folder "./data\_pop/". For Tripadvisor, please put all the hotel review files "hotel\_?????.dat" in the folder "./data\_tripadvisor/Review\_Texts". Then you can

         type in "python test.py -realpop" or "python test.py -realtrip"

If you tpye in "python test.py" in the terminal, you will run the experiments on the default datasets -- synthetic(unimodal). Then you can open the folder "./measure_syn" to see the results.

Specific implementation of our KDEm is included in "KDEm.py".
