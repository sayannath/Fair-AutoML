#!/bin/bash

cd .. || exit
cd evaluation/adult || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done

cd .. || exit
cd bank || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done

cd .. || exit
cd german || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done

cd .. || exit
cd compas || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done

cd .. || exit
cd titanic || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done