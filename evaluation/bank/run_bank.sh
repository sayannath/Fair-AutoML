#!/bin/zsh

# MLP Classifier for Adult dataset
echo "Running MLP SPD age"
python bank_mlp_SPD_age.py >> ./training_logs/bank_mlp_SPD_age.txt

echo "Running MLP EOD age"
python bank_mlp_EOD_age.py  >> ./training_logs/bank_mlp_EOD_age.txt

echo "Running MLP AOD age"
python bank_mlp_AOD_age.py  >> ./training_logs/bank_mlp_AOD_age.txt

# XGB Classifier for Adult dataset
echo "Running XGB SPD age"
python bank_xgb_SPD_age.py >> ./training_logs/bank_xgb_SPD_age.txt

echo "Running XGB EOD age"
python bank_xgb_EOD_age.py  >> ./training_logs/bank_xgb_EOD_age.txt

echo "Running XGB AOD age"
python bank_xgb_AOD_age.py  >> ./training_logs/bank_xgb_AOD_age.txt

# LRG Classifier for Adult dataset
echo "Running LRG SPD age"
python bank_lrg_SPD_age.py >> ./training_logs/bank_lrg_SPD_age.txt

echo "Running LRG EOD age"
python bank_lrg_EOD_age.py  >> ./training_logs/bank_lrg_EOD_age.txt

echo "Running LRG AOD age"
python bank_lrg_AOD_age.py  >> ./training_logs/bank_lrg_AOD_age.txt

# GBC Classifier for Adult dataset
echo "Running GBC SPD age"
python bank_gbc_SPD_age.py >> ./training_logs/bank_gbc_SPD_age.txt

echo "Running GBC EOD age"
python bank_gbc_EOD_age.py  >> ./training_logs/bank_gbc_EOD_age.txt

echo "Running GBC AOD age"
python bank_gbc_AOD_age.py  >> ./training_logs/bank_gbc_AOD_age.txt