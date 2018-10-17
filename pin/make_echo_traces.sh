#!/bin/bash

echo "deleting echo_trace directory"
rm -rf echo_trace
echo "making echo_trace directory"
mkdir -p echo_trace

for size in 1 10 100 1000; do
	for i in `seq 1 25`; do
		# adapted from https://gist.github.com/earthgecko/3089509
		STRING=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w "${size}" | head -n 1)
		"${PIN_ROOT}"/pin -injection child -t obj-intel64/branchtrace.so -- echo "${STRING}"
		# move the output to a specific place
		mv "branchtrace.out" "echo_trace/size_${size}_trace_${i}.out"
	done
done
