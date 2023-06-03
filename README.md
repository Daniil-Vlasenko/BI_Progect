# BI_Progect "Classification of brain activity using Synolitic networks"
Download docs/_build/html and open docs/_build/html/index.html to read the documentation of the code.

## Propose of the project
By now, methods of representing fMRI data in the form of graphs have already been used in the tasks of classifying brain activity. But rather simple classical methods, such as Pearson's linear correlation or partial correlation, are used to reflect relationships between brain regions in such graphs. Although these methods reveal the correlation of features, such correlation, by itself, does not provide useful information for classification. In this paper, we propose to eliminate this drawback and refine the methods of data representation. Thus, the work of this paper is to propose and test a method of representing fMRI data in the form of graphs that would reflect information meaningful for subsequent classification about the relationships between brain regions. We have named this method synolithic, as it was inspired by synolithic networks, which allow the application of graph analysis methods to multidimensional complex data. Here we will consider the simplest case, in which we will distinguish between only two brain modes, or, in other words, when the task is binary classification.

## Result of the project
The method was implemented and tested with the following data. There are two modes in which the subject's brain can function. In the first mode, the subject sees 55 blocks sequentially, 50 of which are different images, 5 of which repeat the previous picture. If the subject sees a repeating picture, he must press the button. This is done to keep the subject's attention. In the second mode, the subject is asked to imagen 25 objects sequentially. After each object is imagined, the subject rates on a five-point scale the degree of clarity of the image he or she imagined by pressing the buttons. Five subjects participated in the data collection. Thus, 24 fMRIs in visual perception mode and 20 fMRIs in memory-based imaging mode were taken from each subject. The sample was divided so that 30% of the sample of each mode fell into the test sample and 70% fell into the training sample. We got an accuracy that is equal to 98.5%. The final classification was based on properties of the graphs. The matrix bellow is a classification matrix that shows results of the classification.

|          | seen | imagined |
|----------|------|----------|
| seen     | 36   | 0        |
| imagined | 1    | 29       |

The figure below shows distributions of graph properties for the entire fMRI sample when. 

[properties_T_median_w_0.pdf](https://github.com/Daniil-Vlasenko/BI_Progect/files/11522409/properties_T_median_w_0.pdf)

So method of representing fMRI data in the form of graphs that would reflect information meaningful for subsequent classification about the relationships between brain regions has been implemented and tested. It is based on Synolitic networks. And it was proved that Synolitic networks are useful for studying brain activity.

## References
1.	Ogawa S, Lee TM, Kay AR, and Tank DW. Brain magnetic resonance imaging with contrast dependent on blood oxygenation. Proceedings of the National Academy of Sciences of the United States of America, 1990; 87(24):9868–9872. DOI: 10.1073/pnas.87.24.9868. 
2.	Singleton MJ. Functional Magnetic Resonance Imaging. Yale J Biol Med. 2009;82(4):233. 
3. 	Gao J, Huth A, Lescroart M, Gallant J. Pycortex: An interactive surface visualizer for fMRI. Frontiers in Neuroinformatics, 2015; 9. DOI: 10.3389/fninf.2015.00023. 
4. 	Li X, Dvornek NC, Zhou Y, Zhuang J, Ventola P, Duncan JS. Graph Neural Network for Interpreting Task-fMRI Biomarkers. Medical Image Computing and Computer Assisted Intervention, 2019; 11768:485-493. DOI: 10.1007/978-3-030-32254-0_54. 
5. 	Saueressig C, Berkley A, Munbodh R, Singh R. A Joint Graph and Image Convolution Network for Automatic Brain Tumor Segmentation. Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. 2021; 12962:356–365. DOI: 10.1007/978-3-031-08999- 2_30. 
6. 	Anderson A and Cohen MS. Decreased small-world functional network connectivity and clustering across resting state networks in schizophrenia: an fMRI classification tutorial. Frontiers in Human Neuroscience, 2013; 7:520. DOI: 10.3389/fnhum.2013.00520. 
7. 	Kim B-H and Ye JC. Understanding Graph Isomorphism Network for rs-fMRI Functional Connectivity Analysis. Frontiers in Neuroscience. 2020; 14:630. DOI: 10.3389/fnins.2020.00630. 
8. 	Nazarenko T, Whitwell HJ, Blyuss O, Zaikin A. Parenclitic and Synolytic Networks Revisited. Frontiers in Genetics. 2021; 12. DOI: 10.3389/fgene.2021.733783. 
9. 	Horikawa T, Kamitani Y. Generic Object Decoding (fMRI on ImageNet). OpenNeuro. 2019. DOI: 10.18112/openneuro.ds001246.v1.2.1. 
10. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011; 12:2825-2830.

