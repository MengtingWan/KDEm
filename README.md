This repository includes data and code for following paper:

Mengting Wan, Xiangyu Chen, Lance Kaplan, Jiawei Han, Jing Gao, Bo Zhao, "From Truth Discovery to Trustworthy Opinion Discovery: An Uncertainty-Aware Quantitative Modeling Approach", in Proc. of 2016 ACM SIGKDD Conf. on Knowledge Discovery and Data Mining (KDD'16), San Francisco, CA, Aug. 2016

Specifically, the core algorithm **KDEm: Kernel Density Estimation from Multiple Sources** is implemented in **KDEm.py**.

Other baseline methods are implemented based on following papers:

- **TruthFinder (TruthFinder.py):** Xiaoxin Yin, Jiawei Han, and Philip S. Yu, "Truth Discovery with Multiple Conflicting Information Providers onthe Web", in Proc. 2007 ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD'07), San Jose, CA, Aug. 2007.
- **AccuSim (Accu.py):** Xin Luna Dong, Laure Berti-Equille, and Divesh Srivastava. "Integrating conflicting data: the role of source dependence." in Proc. 2009 Int. Conf. on Very Large Data Bases (VLDB'09), Lyon, France, Aug. 2009.
- **GTM (GTM.py):** Bo Zhao and Jiawei Han, "A Probabilistic Model for Estimating Real-Valued Truth from Conflicting Sources", in Proc. of 10th Int. Workshop on Quality in Databases, in conjunction with VLDB 2012 (QDB'12), Istanbul, Turkey, Aug. 2012.
- **CRH (CRH.py):** Qi Li, Yaliang Li, Jing Gao, Bo Zhao, Wei Fan, and Jiawei Han, "Resolving Conflicts in Heterogeneous Data by Truth Discovery and Source Reliability Estimation", in Proc. of 2014 ACM SIGMOD Int. Conf. on Management of Data (SIGMOD'14), Snowbird, UT, June 2014.
- **CATD (CATD.py):** Qi Li, Yaliang Li, Jing Gao, Lu Su, Bo Zhao, Murat Demirbas, Wei Fan, and Jiawei Han, "A Confidence-Aware Approach for Truth Discovery on Long-Tail Data",  PVLDB 8(4): 425-436, 2015  Also, in Proc. 2015 Int. Conf. on Very Large Data Bases (VLDB'15), Kohala Coast, Hawaii, Sept. 2015.

If you have any questions, feel free to contact me at m5wan@ucsd.edu

To run experiments on synthetic datasets -- Synthetic(unimodal) and Synthetic(mix), you can directly type in

         "python test.py synuni" or "python test.py synmix"

Notice that experiments may run for a while.

To run experiments on real-world datasets -- Population(outlier) and Tripadvisor, you must download the Population(outlier) data from this [link](http://cogcomp.cs.illinois.edu/page/resource_view/16) and Tripadvisor data from this [link](http://times.cs.uiuc.edu/~wang296/Data/).

For Population(outlier), please put two files "popAnswerOut.txt" and "pupTuples.txt" in the folder "./data\_pop/". For Tripadvisor, please put all the hotel review files "hotel\_?????.dat" in the folder "./data\_tripadvisor/Review\_Texts". Then you can type in

         "python test.py realpop" or "python test.py realtrip"

If you tpye in "python test.py" in the terminal, you will run the experiments on the default datasets -- synthetic(unimodal). Then you can open the folder "./measure_syn" to see the results.
