cd evaluation/bank || exit

# Find all Python files and run them one by one
for file in *.py; do
    echo "Running $file ..."
    python "$file"
    echo "$file completed."
    echo "---------------------"
done