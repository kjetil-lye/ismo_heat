#!/bin/bash
set -e
export PYTHONPATH=${PYTHONPATH}:$(pwd)/iterative_surrogate_optimization
mkdir ensemble_output
cd ensemble_output
python -m ismo.bin.run_ensemble --script_name submit_heat.py --source_folder heat_chain "$@"
