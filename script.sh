#!/usr/bin/env bash
for i in $(ls assets | grep tif); do
	python contours.py $i
done