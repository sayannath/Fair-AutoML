#!/bin/zsh

# MLP Classifier for Bank dataset
echo "Running MLP SPD race"
python mep_MLP_SPD_race.py >> ./training_logs/mep_MLP_SPD_race.txt # 10:50

echo "Running MLP EOD race"
python mep_MLP_EOD_race.py  >> ./training_logs/mep_MLP_EOD_race.txt

echo "Running MLP AOD race"
python mep_MLP_AOD_race.py  >> ./training_logs/mep_MLP_AOD_race.txt

# LRG Classifier for Bank dataset
echo "Running LRG SPD race"
python mep_LRG_SPD_race.py >> ./training_logs/mep_LRG_SPD_race.txt

echo "Running LRG EOD race"
python mep_LRG_EOD_race.py  >> ./training_logs/mep_LRG_EOD_race.txt # 14:50

echo "Running LRG AOD race"
python mep_LRG_AOD_race.py  >> ./training_logs/mep_LRG_AOD_race.txt 

# RF Classifier for Bank dataset
echo "Running RF SPD race"
python mep_RF_SPD_race.py >> ./training_logs/mep_RF_SPD_race.txt 

echo "Running RF EOD race"
python mep_RF_EOD_race.py  >> ./training_logs/mep_RF_EOD_race.txt

echo "Running RF AOD race"
python mep_RF_AOD_race.py  >> ./training_logs/mep_RF_AOD_race.txt # 18:50