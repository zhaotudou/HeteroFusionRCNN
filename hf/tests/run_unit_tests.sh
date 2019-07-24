#!/usr/bin/env bash

cd "$(dirname "$0")"
cd ../..

echo $PYTHONPATH

echo "Running unit tests in $(pwd)/hf"
coverage run --source hf -m unittest discover -b --pattern "*_test.py"

#coverage report -m
