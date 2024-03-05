#!/usr/bin/env bash

if ! out=$(patch -p1 --forward < $( dirname -- "${BASH_SOURCE[0]}" )/$1)
then
    if echo "$out" | grep -q "Reversed (or previously applied) patch detected!  Skipping patch."
    then
        exit 0
    else
        exit 1
    fi
else
    exit 0
fi
