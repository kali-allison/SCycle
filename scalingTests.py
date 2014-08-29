import commands
import os
import string
import os.path
import time
import datetime as datetime

def buildqsub(nodes,Ny,Nz):
    print "Creating shell script."
    script_name="scalingTest.sh"
    scriptfile = open(script_name, "w")
    scriptfile.write("#!/bin/bash\n")
    scriptfile.write("#PBS -N nodes%u\n" %(nodes)) #name of job
    scriptfile.write("#PBS -l nodes=%u:ppn=%u\n"%(nodes,16) )
    scriptfile.write("#PBS -q tgp\n")
    scriptfile.write("#PBS -V\n")
    scriptfile.write("#PBS -m n\n")
    scriptfile.write("#PBS -k oe\n")
    scriptfile.write("#PBS -e eqCycle_nodes%u.err\n" %nodes)
    scriptfile.write("#PBS -o eqCycle_nodes%u.out\n" %nodes)
    scriptfile.write("#\n")
    scriptfile.write("EXEC_DIR=/data/dunham/kallison/eqCycle\n")
    scriptfile.write("cd $PBS_O_WORKDIR\n")
    scriptfile.write("#\n")

    scriptfile.write("mpirun $EXEC_DIR/main -log_summary\n" %(Ny,Nz) )
    scriptfile.close()

    print "Submitting job to queue."
    jobname = "nodes"+nodes
    stat,out=commands.getstatusoutput("qsub %s"%script_name)
    finished = 0
    interval = 5
    user=os.environ['USER']
    while not finished:
        time.sleep(interval)
        #~finished,out=commands.getstatusoutput("qstat | grep  kallison | grep -v C")
        finished,out=commands.getstatusoutput("qstat -u kallison | grep %s",jobname)
        print out

#strong test
results = open("results.txt",'w')
results.write('---Strong Test: ' + str(time.time())+'\n')
results.write("nodes m n runtime\n")
for power in range(0,3):
    nodes = 2**power
    buildqsub(nodes)

