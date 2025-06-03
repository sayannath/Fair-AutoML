#!/bin/zsh

# XGB Classifier for German Dataset
echo "Running XGB SPD age"
python german_XGB_SPD_age.py >> ./training_logs/german_XGB_SPD_age.txt

echo "Running German XGB EOD age"
python german_XGB_EOD_age.py >> ./training_logs/german_XGB_EOD_age.txt

echo "Running German XGB AOD age"
python german_XGB_AOD_age.py >> ./training_logs/german_XGB_AOD_age.txt

echo "Running german XGB SPD sex"
python german_XGB_SPD_sex.py >> ./training_logs/german_XGB_SPD_sex.txt

echo "Running german XGB EOD sex"
python german_XGB_EOD_sex.py >> ./training_logs/german_XGB_EOD_sex.txt

echo "Running german XGB AOD sex"
python german_XGB_AOD_sex.py >> ./training_logs/german_XGB_AOD_sex.txt

# RandomForest Classifier for German Dataset
echo "Running RF SPD age"
python german_RF_SPD_age.py >> ./training_logs/german_RF_SPD_age.txt

echo "Running RF EOD age"
python german_RF_EOD_age.py >> ./training_logs/german_RF_EOD_age.txt

echo "Running RF AOD age"
python german_RF_AOD_age.py >> ./training_logs/german_RF_AOD_age.txt

echo "Running RF SPD sex"
python german_RF_SPD_sex.py >> ./training_logs/german_RF_SPD_sex.txt

echo "Running RF EOD sex"
python german_RF_EOD_sex.py >> ./training_logs/german_RF_EOD_sex.txt

echo "Running RF AOD sex"
python german_RF_AOD_sex.py >> ./training_logs/german_RF_AOD_sex.txt

# KNN Classifier for German Dataset
echo "Running KNN SPD age"
python german_KNN_SPD_age.py >> ./training_logs/german_KNN_SPD_age.txt

echo "Running KNN EOD age"
python german_KNN_EOD_age.py >> ./training_logs/german_KNN_EOD_age.txt

echo "Running KNN AOD age"
python german_KNN_AOD_age.py >> ./training_logs/german_KNN_AOD_age.txt

echo "Running KNN SPD sex"
python german_KNN_SPD_sex.py >> ./training_logs/german_KNN_SPD_sex.txt

echo "Running KNN EOD sex"
python german_KNN_EOD_sex.py >> ./training_logs/german_KNN_EOD_sex.txt

echo "Running KNN AOD sex"
python german_KNN_AOD_sex.py >> ./training_logs/german_KNN_AOD_sex.txt
