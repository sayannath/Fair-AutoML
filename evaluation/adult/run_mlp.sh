#!/bin/zsh

#python adult_mlp_SPD_race.py
#python adult_mlp_EOD_race.py
#python adult_mlp_AOD_race.py

echo "Running MLP SPD SEX"
python adult_mlp_SPD_sex.py

echo "Running MLP EOD SEX"
python adult_mlp_EOD_sex.py

echo "Running MLP AOD SEX"
python adult_mlp_AOD_sex.py