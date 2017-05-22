#!/bin/bash 

set -e

function makedir()
{
    root_path=$1

    [ ! -d $root_path"/train" ] && mkdir -p $root_path"/train"
    [ ! -d $root_path"/test" ] && mkdir -p $root_path"/test"
    [ ! -d $root_path"/valid" ] && mkdir -p $root_path"/valid"
    [ ! -d $root_path"/results" ] && mkdir -p $root_path"/results"
}

function shuffle_to_valid()
{
    root_path=$1
    shuf_num=$2
    train_path=$1"/train/"
    valid_path=$1"/valid/"
    
    for img in `ls $train_path | shuf -n $shuf_num`; do
        mv $train_path$img $valid_path
    done
}

function shuffle_to_sample()
{
    src_path="./data/"$1"/"
    trg_path="./data/sample/"$1"/"
    sample_num=$2

    for img in `ls $src_path | shuf -n $sample_num`; do
        cp $src_path$img $trg_path
    done
}

function make_subcategories()
{
    _path=$1

    mkdir $_path"/dogs/"
    mv $_path/dog.*.jpg $_path"/dogs/"
    mkdir $_path"/cats/"
    mv $_path/cat.*.jpg $_path"/cats/"
}

path="./data/"
sample_path="./data/sample/"

makedir $path
makedir $sample_path

shuffle_to_valid $path 2000

shuffle_to_sample "train" 200
shuffle_to_sample "test" 50
shuffle_to_sample "valid" 50

make_subcategories $path"/train/"
make_subcategories $path"/valid/"
make_subcategories $sample_path"/train/"
make_subcategories $sample_path"/valid/"

mkdir $path"/test/unknown/"
mv $path/test/*.jpg $path"/test/unknown"

mkdir $sample_path"/test/unknown/"
mv $sample_path/test/*.jpg $sample_path"/test/unknown"

