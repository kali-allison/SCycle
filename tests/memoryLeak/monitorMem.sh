#!/bin/bash
#~ echo $1
qstat -f $1 | grep resources_used.*mem
qstat -f $1 | grep resources_used.walltime
