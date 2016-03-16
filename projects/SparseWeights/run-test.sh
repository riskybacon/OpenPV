#!/bin/sh

threads="1 2 4 6 8 10 12 14 16"
num_samples=3
cmd="
for nt in $threads ; do
   for i in  $(seq 1 $num_samples); do 
      cmd
      echo $i
      time ls
   done
done

#time ~/src/build/PVSystemTests/LCATest/Release/LCATest -p /Users/jbowles/src/LCATest/input/LCATest.params -c checkpoints/Checkpoint6 --testall -t 4 -l /Users/jbowles/src/LCATest/LCATest.log

