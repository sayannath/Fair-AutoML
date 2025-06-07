#!/bin/zsh

## MLP Classifier for Bank dataset
#echo "Running MLP SPD age"
#python bank_MLP_SPD_age.py >> ./training_logs/bank_MLP_SPD_age.txt
#
#echo "Running MLP EOD age"
#python bank_MLP_EOD_age.py  >> ./training_logs/bank_MLP_EOD_age.txt
#
#echo "Running MLP AOD age"
#python bank_MLP_AOD_age.py  >> ./training_logs/bank_MLP_AOD_age.txt
#
## XGB Classifier for Bank dataset
#echo "Running XGB SPD age"
#python bank_XGB_SPD_age.py >> ./training_logs/bank_XGB_SPD_age.txt
#
#echo "Running XGB EOD age"
#python bank_XGB_EOD_age.py  >> ./training_logs/bank_XGB_EOD_age.txt
#
#echo "Running XGB AOD age"
#python bank_XGB_AOD_age.py  >> ./training_logs/bank_XGB_AOD_age.txt

## LRG Classifier for Bank dataset
#echo "Running LRG SPD age"
#python bank_LRG_SPD_age.py >> ./training_logs/bank_LRG_SPD_age.txt
#
#echo "Running LRG EOD age"
#python bank_LRG_EOD_age.py  >> ./training_logs/bank_LRG_EOD_age.txt
#
#echo "Running LRG AOD age"
#python bank_LRG_AOD_age.py  >> ./training_logs/bank_LRG_AOD_age.txt # 1:00

## GBC Classifier for Bank dataset
#echo "Running GBC SPD age"
#python bank_GBC_SPD_age.py >> ./training_logs/bank_GBC_SPD_age.txt
#
#echo "Running GBC EOD age"
#python bank_GBC_EOD_age.py  >> ./training_logs/bank_GBC_EOD_age.txt
#
#echo "Running GBC AOD age"
#python bank_GBC_AOD_age.py  >> ./training_logs/bank_GBC_AOD_age.txt

## RF Classifier for Bank dataset
#echo "Running RF SPD age"
#python bank_RF_SPD_age.py >> ./training_logs/bank_RF_SPD_age.txt # 2:00
#
#echo "Running RF EOD age"
#python bank_RF_EOD_age.py  >> ./training_logs/bank_RF_EOD_age.txt # 3:00

echo "Running RF AOD age"
python bank_RF_AOD_age.py  >> ./training_logs/bank_RF_AOD_age.txt # 20:49 - 21:49

cd ..

cd german/ || exit
./run_german.sh