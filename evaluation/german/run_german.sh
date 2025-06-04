#!/bin/zsh

echo "German Dataset"

## XGB Classifier for German Dataset
#echo "Running XGB SPD age"
#python german_XGB_SPD_age.py >> ./training_logs/german_XGB_SPD_age.txt
#
#echo "Running XGB EOD age"
#python german_XGB_EOD_age.py >> ./training_logs/german_XGB_EOD_age.txt
#
#echo "Running XGB AOD age"
#python german_XGB_AOD_age.py >> ./training_logs/german_XGB_AOD_age.txt
#
#echo "Running XGB SPD sex"
#python german_XGB_SPD_sex.py >> ./training_logs/german_XGB_SPD_sex.txt
#
#echo "Running XGB EOD sex"
#python german_XGB_EOD_sex.py >> ./training_logs/german_XGB_EOD_sex.txt
#
#echo "Running XGB AOD sex"
#python german_XGB_AOD_sex.py >> ./training_logs/german_XGB_AOD_sex.txt
#
## RandomForest Classifier for German Dataset
#echo "Running RF SPD age"
#python german_RF_SPD_age.py >> ./training_logs/german_RF_SPD_age.txt
#
#echo "Running RF EOD age"
#python german_RF_EOD_age.py >> ./training_logs/german_RF_EOD_age.txt
#
#echo "Running RF AOD age"
#python german_RF_AOD_age.py >> ./training_logs/german_RF_AOD_age.txt
#
#echo "Running RF SPD sex"
#python german_RF_SPD_sex.py >> ./training_logs/german_RF_SPD_sex.txt
#
#echo "Running RF EOD sex"
#python german_RF_EOD_sex.py >> ./training_logs/german_RF_EOD_sex.txt
#
#echo "Running RF AOD sex"
#python german_RF_AOD_sex.py >> ./training_logs/german_RF_AOD_sex.txt
#
## KNN Classifier for German Dataset
#echo "Running KNN SPD age"
#python german_KNN_SPD_age.py >> ./training_logs/german_KNN_SPD_age.txt
#
#echo "Running KNN EOD age"
#python german_KNN_EOD_age.py >> ./training_logs/german_KNN_EOD_age.txt
#
#echo "Running KNN AOD age"
#python german_KNN_AOD_age.py >> ./training_logs/german_KNN_AOD_age.txt
#
#echo "Running KNN SPD sex"
#python german_KNN_SPD_sex.py >> ./training_logs/german_KNN_SPD_sex.txt
#
#echo "Running KNN EOD sex"
#python german_KNN_EOD_sex.py >> ./training_logs/german_KNN_EOD_sex.txt
#
#echo "Running KNN AOD sex"
#python german_KNN_AOD_sex.py >> ./training_logs/german_KNN_AOD_sex.txt
#
## SVC Classifier for German Dataset
#echo "Running SVC SPD age"
#python german_SVC_SPD_age.py >> ./training_logs/german_SVC_SPD_age.txt
#
#echo "Running SVC EOD age"
#python german_SVC_EOD_age.py >> ./training_logs/german_SVC_EOD_age.txt
#
#echo "Running SVC AOD age"
#python german_SVC_AOD_age.py >> ./training_logs/german_SVC_AOD_age.txt
#
#echo "Running SVC SPD sex"
#python german_SVC_SPD_sex.py >> ./training_logs/german_SVC_SPD_sex.txt
#
#echo "Running SVC EOD sex"
#python german_SVC_EOD_sex.py >> ./training_logs/german_SVC_EOD_sex.txt
#
#echo "Running SVC AOD sex"
#python german_SVC_AOD_sex.py >> ./training_logs/german_SVC_AOD_sex.txt

# LRG Classifier for German Dataset
echo "Running LRG SPD age"
python german_LRG_SPD_age.py >> ./training_logs/german_LRG_SPD_age.txt # Needs a re-run

#echo "Running LRG EOD age"
#python german_LRG_EOD_age.py >> ./training_logs/german_LRG_EOD_age.txt
#
#echo "Running LRG AOD age"
#python german_LRG_AOD_age.py >> ./training_logs/german_LRG_AOD_age.txt
#
#echo "Running LRG SPD sex"
#python german_LRG_SPD_sex.py >> ./training_logs/german_LRG_SPD_sex.txt
#
#echo "Running LRG EOD sex"
#python german_LRG_EOD_sex.py >> ./training_logs/german_LRG_EOD_sex.txt
#
#echo "Running LRG AOD sex"
#python german_LRG_AOD_sex.py >> ./training_logs/german_LRG_AOD_sex.txt
#
## MLP Classifier for German Dataset
#echo "Running MLP SPD age"
#python german_MLP_SPD_age.py >> ./training_logs/german_MLP_SPD_age.txt
#
#echo "Running MLP EOD age"
#python german_MLP_EOD_age.py >> ./training_logs/german_MLP_EOD_age.txt

echo "Running MLP AOD age"
python german_MLP_AOD_age.py >> ./training_logs/german_MLP_AOD_age.txt

echo "Running MLP SPD sex"
python german_MLP_SPD_sex.py >> ./training_logs/german_MLP_SPD_sex.txt

echo "Running MLP EOD sex"
python german_MLP_EOD_sex.py >> ./training_logs/german_MLP_EOD_sex.txt

echo "Running MLP AOD sex"
python german_MLP_AOD_sex.py >> ./training_logs/german_MLP_AOD_sex.txt

cd ..

cd compas/
./run_compas.sh