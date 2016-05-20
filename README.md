To run the experiments on datasets, you can 

         open test.py and click "Run", 
         
or

         type in "python test.py" in the terminal
         
Then you can see the results in output file. 

To run experiments on synthetic datasets -- Synthetic(unimodal) and Synthetic(mix), you can directly 

         type in "python test.py -synuni" or "python test.py -synmix"

Notice that experiments may run for a while.

To run experiments on real-world datasets -- Population(outlier) and Tripadvisor, you must download the Population(outlier) data from this [link](http://cogcomp.cs.illinois.edu/page/resource_view/16) and Tripadvisor data from this [link](http://times.cs.uiuc.edu/~wang296/Data/).

For Population(outlier), please put two files "popAnswerOut.txt" and "pupTuples.txt" in the folder "./data\_pop/". For Tripadvisor, please put all the hotel review files "hotel\_?????.dat" in the folder "./data\_tripadvisor/Review\_Texts". Then you can

         type in "python test.py -realpop" or "python test.py -realtrip"

If you tpye in "python test.py" in the terminal, you will run the experiments on the default datasets -- synthetic(unimodal). Then you can open the folder "./measure_syn" to see the results.

Specific implementation of our KDEm is included in "KDEm.py".