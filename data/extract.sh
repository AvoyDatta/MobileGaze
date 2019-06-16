#!/bin/bash
# Extracts GazeCapture data from .tar file in same directory, 
# into a data folder one level out

#Extracts .tar.gz to current dir
tar -xvf gazecapture.tar

#Rm .tar
rm gazecapture.tar

#Extracts .tar.gz to current dir
for a in `ls -1 *.tar.gz`; 
do 
tar -zxvf $a -C .; 
done

#Remove all .tar.gz
rm *.tar.gz
