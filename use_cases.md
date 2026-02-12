User story #1:

Walter works in molecule discovery at Amyris. They want to propose new sesquiterpenes with specific heats of combustion for use as biofuels. They hope to identify new sesquiterpenes with good diversity. They have an MS in natural product chemistry.

Use case #1:
-User downloads inference repo from github.
-User loads pre-trained model from Hugging Face Hub using transformers lib.
-User specifies the type of molecule they want to generate, along with any additional conditioning information such as qed drug likeness or SA accessibility score. 
-User runs inference through command line, leveraging wrapper script using generate function provided through transformers library to run inference, pre-filter for valid, novel, and unique outputs.
-Filter for natural products consistent with NPClassifier label. 
-Provides txt file of outputs for user, and optionally SDF as user provided flag and summary report.



