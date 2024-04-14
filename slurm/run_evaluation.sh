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
cd evaluation/bank || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done

cd .. || exit
cd evaluation/german || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done

cd .. || exit
cd evaluation/compas || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done

cd .. || exit
cd evaluation/titanic || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done