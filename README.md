# Virtual Rat
##### Virtual Rat project is a computaional neuroscience project  conducted by Xiangci Li as a research assistant at Erlich lab under the guidance of New York University Shanghai Professor Jeffrey Erlich. We showed that the task switch cost of rat's ProAnti orienting task can be modeled by an Elman Recurrent Neural Network. The experimental data are from Dr. Carlos Brody's lab at Princeton.

There is a [video](https://www.youtube.com/watch?v=Xs9DYRXQxNg&t=1s) available for this project!

For any question, please contact
##### Xiangci Li (xiangci.li@nyu.edu)
##### Jeffrey Erlich (jerlich@nyu.edu)

### Briefly about the implementation
From the perspective of implementation, Virtual Rat project consists of two parts: Virtual Rat (RNN model) and data analysis of rats' behavioral and electrophysiology data.

### How to reproduce all results:

## Environment setup: 

### Python:

1. Visit https://github.com/dmlc/minpy to setup Minpy and MXNet.
2. Other requirements: Python 2.7, Numpy, Matplotlib, Scikit-learn, Scipy

### MATLAB: 
MATLAB 2017a or later.

## Note:
The codes for the final version of this project are all under `publication`  folder.

`figures`  folder saves figures produced. `mats` folder saves `.mat` files and `pkls` folder  saves `.pkl` files.

## Procedures:

### Virtual Rat part: 

#### 1. Experiments for the most basic RNNs:
   1. Run `sbatch trainingTime.sh` on HPC to evoke `TrainingTime.py`. All weights after certain training epochs will be saved.
   2. Run `sbatch testTrainingTime.sh` on HPC to evoke `TestTrainingTime.py`. Similarly run `sbatch testTrainingTimeFine.sh` to evoke `TestTrainingTimeFine.py`. The weights saved will be loaded to test the performance of the models and the results will be saved in the folder `TrainingTime`.
   3. Run jupyter notebook `TrainingTime.ipynb`. This notebook finds out best epochs to plot later figures.
   4. Run jupyter notebook `VirtualRat.ipynb` and `DiluteActivation.ipynb`to obtain figures. 
   5. Run jupyter notebook `PETH.ipynb` to choose sample PETH.

#### 2. Experiments for varying block sizes:
   1. Run `sbatch trainBlock.sh` on HPC to evoke `BlockTime.py`. All weights after certain training epochs will be saved.
   2. Run `sbatch testBlockTime.sh` on HPC to evoke `TestBlockTime.py`.  The weights saved will be loaded to test the performance of the models and the results will be saved in the folder `BlockTime`.
   3. Run jupyter notebook `Block.ipynb`.

#### 3. Experiments for varying pro to anti ratio:
   1. Run `sbatch RatioTime.sh` on HPC to evoke `TrainRatioTime.py`. All weights after certain training epochs will be saved.
   2. Run `sbatch testTrainingTimeRatio.sh` on HPC to evoke `TestTrainingTimeRatio.py`.  The weights saved will be loaded to test the performance of the models and the results will be saved in the folder `RatioTime`.
   3. Run jupyter notebook `TrainingTimeRatio.ipynb` to check the performace over time.
   4. Run jupyter notebook `Ratio.ipynb`.


### Data analysis part (MATLAB & Python):

1. Run `convertBehavior.m` to extract necessary information from the original huge data file for running Python scripts.
2. Run `countSpikes.m` to produce spike counts.
3. Run `realRatEphysTrainClfs.py` and `realRatEphysTrainClfsTarget.py`
4. Run `ratSGDrule.m`,  `ratSGDTarget.m`,  `RNN_SGD_rule.m` and `RNN_SGD_target.m`.
5. Run jupyter notebook `realRatEphys.ipynb`.
6. Run jupyter notebook `realRatEphysTarget.ipynb`.	
7. Run jupyter notebook `RNNEphys.ipynb`.
8. Run jupyter notebook `realRatBehavior.ipynb`.

## Documentation to files, funtions and data structures
### Data analysis part
#### convertBehavior.m
* Collect pro anti and other data from original spike data
  * Saved in `SessionInfo.mat`, a cell type structure.
  * Each cell has double of (N_of_trials, 4)
  * Each trial has (pro, target_right, switch, hit)
* Collect brain area data
  * Saved in `BrainArea.mat`, a cell type structure.

  * Each cell has double of (1, N_of_cells)

  * Index: brain area
    ​    0: left mPFC
    ​    1: right mPFC
    ​    2: left SC
    ​    3: right SC
    ​    4: left FOF
    ​    5: right FOF
* Collect rat index

  * Saved in `RatIndexPerSession.mat`, (N_session, 1) double. 
* Collect cellid per session
  * Saved in `CellIndexPerSession.mat`, a cell type structure.
  * Each cell has double of (1, N_of_cells), containing cell indices (not the raw cell ID, but still sufficient)
* Collect Duan's session index and Erlich's session indices.
#### countSpikes.m
* Count spikes from Duan's ephys data and Erlich's ephys data using different time periods and time steps.
  * Duan's spike count includes delay step, which matches the RNN model.
  * Erlich's does not include delay step.
* Saved in `SpikeCountsPerSessionDuan.mat` and `SpikeCountsPerSessionErlich.mat`, cell type structure.
* Each cell is a double with shape (5 steps, num_cells recorded in this session, num_trials)
#### singleCellRuleEncoding.m
* Separately calculate for Erlich's and Duan's data.
* Save as `Duan_single_AUC_p.mat`, an array with (N_cells,10)
* Each column corresponds to 'ITI_auc','ITI_p','rule_auc','rule_p','delay_auc','delay_p','target_auc','target_p','choice_auc','choice_p'

#### realRatEphysTrainClfs.py
* This script trains (overfit) logistic regression classifier to predict each trials' rule 
encoding score and save them in to an array which will be converted to MATLAB table for fitting GLME.
* Saves results computed in `experimentor_SGD_table_brainArea.mat`
	* 21 categories: session_index, pro, right, switch, hit, rat_index, score0, score1, score2, score3, score4,accuracy0, accuracy1, accuracy2, accuracy3, accuracy4, encoding0, encoding1, encoding2, encoding3, encoding4
