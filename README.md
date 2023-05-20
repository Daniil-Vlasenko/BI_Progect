# BI_Progect "Classification of brain activity using Synolitic networks"

## Propose of the project
By now, methods of representing fMRI data in the form of graphs have already been used in the tasks of classifying brain activity. But rather simple classical methods, such as Pearson's linear correlation or partial correlation, are used to reflect relationships between brain regions in such graphs. Although these methods reveal the correlation of features, such correlation, by itself, does not provide useful information for classification. In this paper, we propose to eliminate this drawback and refine the methods of data representation. Thus, the work of this paper is to propose and test a method of representing fMRI data in the form of graphs that would reflect information meaningful for subsequent classification about the relationships between brain regions. We have named this method synolithic, as it was inspired by synolithic networks, which allow the application of graph analysis methods to multidimensional complex data. Here we will consider the simplest case, in which we will distinguish between only two brain modes, or, in other words, when the task is binary classification.

## Result of the project
The method was implemented and tested with the following data. There are two modes in which the subject's brain can function. In the first mode, the subject sees 55 blocks sequentially, 50 of which are different images, 5 of which repeat the previous picture. If the subject sees a repeating picture, he must press the button. This is done to keep the subject's attention. In the second mode, the subject is asked to imagen 25 objects sequentially. After each object is imagined, the subject rates on a five-point scale the degree of clarity of the image he or she imagined by pressing the buttons. Five subjects participated in the data collection. Thus, 24 fMRIs in visual perception mode and 20 fMRIs in memory-based imaging mode were taken from each subject. The sample was divided so that 30% of the sample of each mode fell into the test sample and 70% fell into the training sample. We got an accuracy that is equal to 98.5%.

The final classification was based on properties of the graphs. The figure below shows distributions of graph properties for the entire fMRI sample when.

[properties_T_median_w_0.pdf](https://github.com/Daniil-Vlasenko/BI_Progect/files/11522409/properties_T_median_w_0.pdf)

So method of representing fMRI data in the form of graphs that would reflect information meaningful for subsequent classification about the relationships between brain regions has been implemented and tested. It is based on Synolitic networks. And it was proved that Synolitic networks are useful for studying brain activity.



