#!/usr/bin/env bash

files = ("$1/*")

function join_by { local IFS="$1"; shift; echo "$*"; }

videos = join_by "," "${arr[@]}"
echo "Going to run VIBE on video ${videos}"

python demo.py --videos videos --output_folder output/ --staf_dir ./openpose
