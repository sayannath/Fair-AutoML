#!/bin/zsh

## MLP Classifier for Adult dataset
#echo "Running MLP SPD RACE"
#python adult_mlp_SPD_race.py >> ./training_logs/adult_mlp_SPD_race.txt
#
#echo "Running MLP EOD RACE"
#python adult_mlp_EOD_race.py  >> ./training_logs/adult_mlp_EOD_race.txt
#
#echo "Running MLP AOD RACE"
#python adult_mlp_AOD_race.py  >> ./training_logs/adult_mlp_AOD_race.txt
#
#echo "Running MLP SPD SEX"
#python adult_mlp_SPD_sex.py  >> ./training_logs/adult_mlp_SPD_sex.txt
#
#echo "Running MLP EOD SEX"
#python adult_mlp_EOD_sex.py  >> ./training_logs/adult_mlp_EOD_sex.txt
#
#echo "Running MLP AOD SEX"
#python adult_mlp_AOD_sex.py  >> ./training_logs/adult_mlp_AOD_sex.txt
#
## XGB Classifier for Adult dataset
#echo "Running XGB SPD RACE"
#python adult_xgb_SPD_race.py >> ./training_logs/adult_xgb_SPD_race.txt
#
#echo "Running XGB EOD RACE"
#python adult_xgb_EOD_race.py  >> ./training_logs/adult_xgb_EOD_race.txt
#
#echo "Running XGB AOD RACE"
#python adult_xgb_AOD_race.py  >> ./training_logs/adult_xgb_AOD_race.txt
#
#echo "Running XGB SPD SEX"
#python adult_xgb_SPD_sex.py  >> ./training_logs/adult_xgb_SPD_sex.txt
#
#echo "Running XGB EOD SEX"
#python adult_xgb_EOD_sex.py  >> ./training_logs/adult_xgb_EOD_sex.txt
#
#echo "Running XGB AOD SEX"
#python adult_xgb_AOD_sex.py  >> ./training_logs/adult_xgb_AOD_sex.txt
#
## LRG Classifier for Adult dataset
#echo "Running LRG SPD RACE"
#python adult_LRG_SPD_race.py >> ./training_logs/adult_lrg_SPD_race.txt

echo "Running LRG EOD RACE"
python adult_LRG_EOD_race.py  >> ./training_logs/adult_LRG_EOD_race.txt

echo "Running LRG AOD RACE"
python adult_LRG_AOD_race.py  >> ./training_logs/adult_LRG_AOD_race.txt

#echo "Running LRG SPD SEX"
#python adult_LRG_SPD_sex.py  >> ./training_logs/adult_lrg_SPD_sex.txt

echo "Running LRG EOD SEX"
python adult_LRG_EOD_sex.py  >> ./training_logs/adult_LRG_EOD_sex.txt

echo "Running LRG AOD SEX"
python adult_LRG_AOD_sex.py  >> ./training_logs/adult_LRG_AOD_sex.txt

# # GBC Classifier for Adult dataset
# echo "Running GBC SPD RACE"
# python adult_gbc_SPD_race.py >> ./training_logs/adult_gbc_SPD_race.txt

# echo "Running GBC EOD RACE"
# python adult_gbc_EOD_race.py  >> ./training_logs/adult_gbc_EOD_race.txt

# echo "Running GBC AOD RACE"
# python adult_gbc_AOD_race.py  >> ./training_logs/adult_gbc_AOD_race.txt

# echo "Running GBC SPD SEX"
# python adult_gbc_SPD_sex.py  >> ./training_logs/adult_gbc_SPD_sex.txt

# echo "Running GBC EOD SEX"
# python adult_gbc_EOD_sex.py  >> ./training_logs/adult_gbc_EOD_sex.txt

# echo "Running GBC AOD SEX"
# python adult_gbc_AOD_sex.py  >> ./training_logs/adult_gbc_AOD_sex.txt

# RF Classifier for Adult dataset
echo "Running RF SPD RACE"
python adult_RF_SPD_race.py >> ./training_logs/adult_RF_SPD_race.txt

echo "Running RF EOD RACE"
python adult_RF_EOD_race.py  >> ./training_logs/adult_RF_EOD_race.txt

echo "Running RF AOD RACE"
python adult_RF_AOD_race.py  >> ./training_logs/adult_RF_AOD_race.txt

echo "Running RF SPD SEX"
python adult_RF_SPD_sex.py  >> ./training_logs/adult_RF_SPD_sex.txt

echo "Running RF EOD SEX"
python adult_RF_EOD_sex.py  >> ./training_logs/adult_RF_EOD_sex.txt

echo "Running RF AOD SEX"
python adult_RF_AOD_sex.py  >> ./training_logs/adult_RF_AOD_sex.txt

cd ..
cd bank/

./run_bank.sh