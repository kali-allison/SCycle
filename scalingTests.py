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
    scriptfile.write("#PBS -N nodes%u_Ny%u_Nz%u\n" %(nodes,Ny,Nz)) #name of job
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

    scriptfile.write("mpirun $EXEC_DIR/main -order 4 -Ny %u -Nz %u -log_summary\n" %(Ny,Nz) )
    scriptfile.close()

    print "Submitting job to queue."
    stat,out=commands.getstatusoutput("qsub %s"%script_name)
    finished = 0
    interval = 5
    user=os.environ['USER']
    while not finished:
        time.sleep(interval)
        finished,out=commands.getstatusoutput("qstat | grep  kallison | grep -v C")
        print out

#strong test
results = open("results.txt",'w')
results.write('---Strong Test: ' + str(time.time())+'\n')
results.write("nodes m n runtime\n")
for nodes in range(1,2):
    Ny=1201
    Nz=401
    buildqsub(nodes,Ny,Nz)

