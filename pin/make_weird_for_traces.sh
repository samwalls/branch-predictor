#!/bin/bash

echo "deleting weird_for_trace directory"
rm -rf weird_for_trace
echo "making weird_for_trace directory"
mkdir -p weird_for_trace

# recompile the weird_for program
rm -f weird_for
gcc src/weird_for.c -o weird_for
chmod +x weird_for

for i in `seq 1 5`; do
		"${PIN_ROOT}"/pin -injection child -t obj-intel64/branchtrace.so -- ./weird_for
		# move the output to a specific place
		mv "branchtrace.out" "weird_for_trace/trace_${i}.out"
done
# generate a subset files for use in profiled predictors
head -n 500 "weird_for_trace/trace_1.out" > "weird_for_trace/500_head.out"
head -n 400 "weird_for_trace/trace_2.out" > "weird_for_trace/400_head.out"
head -n 300 "weird_for_trace/trace_1.out" > "weird_for_trace/300_head.out"
head -n 200 "weird_for_trace/trace_1.out" > "weird_for_trace/200_head.out"
head -n 100 "weird_for_trace/trace_1.out" > "weird_for_trace/100_head.out"

