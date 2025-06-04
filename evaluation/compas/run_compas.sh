#!/bin/zsh

echo "COMPAS Dataset"

# RandomForest Classifier for COMPAS Dataset
echo "Running RF SPD race"
python compas_RF_SPD_race.py >> ./training_logs/compas_RF_SPD_race.txt

echo "Running RF EOD race"
python compas_RF_EOD_race.py >> ./training_logs/compas_RF_EOD_race.txt

echo "Running RF AOD race"
python compas_RF_AOD_race.py >> ./training_logs/compas_RF_AOD_race.txt

echo "Running RF SPD sex"
python compas_RF_SPD_sex.py >> ./training_logs/compas_RF_SPD_sex.txt

echo "Running RF EOD sex"
python compas_RF_EOD_sex.py >> ./training_logs/compas_RF_EOD_sex.txt

echo "Running RF AOD sex"
python compas_RF_AOD_sex.py >> ./training_logs/compas_RF_AOD_sex.txt

# LRG Classifier for COMPAS Dataset
echo "Running LRG SPD race"
python compas_LRG_SPD_race.py >> ./training_logs/compas_LRG_SPD_race.txt

echo "Running LRG EOD race"
python compas_LRG_EOD_race.py >> ./training_logs/compas_LRG_EOD_race.txt

echo "Running LRG AOD race"
python compas_LRG_AOD_race.py >> ./training_logs/compas_LRG_AOD_race.txt

echo "Running LRG SPD sex"
python compas_LRG_SPD_sex.py >> ./training_logs/compas_LRG_SPD_sex.txt

echo "Running LRG EOD sex"
python compas_LRG_EOD_sex.py >> ./training_logs/compas_LRG_EOD_sex.txt

echo "Running LRG AOD sex"
python compas_LRG_AOD_sex.py >> ./training_logs/compas_LRG_AOD_sex.txt

# MLP Classifier for German Dataset
echo "Running MLP SPD race"
python compas_MLP_SPD_race.py >> ./training_logs/compas_MLP_SPD_race.txt

echo "Running MLP EOD race"
python compas_MLP_EOD_race.py >> ./training_logs/compas_MLP_EOD_race.txt

echo "Running MLP AOD race"
python compas_MLP_AOD_race.py >> ./training_logs/compas_MLP_AOD_race.txt

echo "Running MLP SPD sex"
python compas_MLP_SPD_sex.py >> ./training_logs/compas_MLP_SPD_sex.txt

echo "Running MLP EOD sex"
python compas_MLP_EOD_sex.py >> ./training_logs/compas_MLP_EOD_sex.txt

echo "Running MLP AOD sex"
python compas_MLP_AOD_sex.py >> ./training_logs/compas_MLP_AOD_sex.txt