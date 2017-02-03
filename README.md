# AXA Data Challenge

Data Challenge provided by AXA in november 2016 at École Polytechnique, in which I ended up finishing 1st with my team.
The purpose is to predict the number of incoming calls for the AXA call center in France, on a per "half-hour" time slot basis.

## HOW TO USE

### In data/ directory:
* Join the compressed files: 
<pre><code> cat train_compressed.tar.gz.a* > train_compressed.tar.gz </code></pre>
* Extract the data:
<pre><code> tar zxvf train_compressed.tar.gz </code></pre>

### Run the pipeline:
* Execute features_selection.py to clear the data, it outputs a new file called data_reduced.csv
* Execute feature_engeneering.py to proceed to data transformation, it outputs a new file called data_transformed.csv
* Execute main.py to generate the predictions

### Potentially, you can launch a cross validation by running the cross_validation.py file

