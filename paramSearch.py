import commands
import os
import string
import os.path
import datetime
import time
import math
import datetime as datetime
import subprocess as sb

def buildLaunchScript(mu,initFile,baseDir):

  jobname = "mu_" + str(mu)

  #~print "Creating shell script."
  scriptName = baseDir + "/paramSearch.sh"
  scriptfile = open(scriptName, "w")
  scriptfile.write("#!/bin/bash\n")
  scriptfile.write("#PBS -N %s\n" %jobname) #name of job
  scriptfile.write("#PBS -l nodes=%u:ppn=%u\n"%(4,16) )
  scriptfile.write("#PBS -q tgp\n")
  scriptfile.write("#PBS -V\n")
  scriptfile.write("#PBS -m n\n")
  scriptfile.write("#PBS -k oe\n")
  scriptfile.write("#PBS -e %s/ss_mu_%u.err\n" %(baseDir,mu))
  scriptfile.write("#PBS -o %s/ss_mu_%u.out\n" %(baseDir,mu))
  scriptfile.write("#\n")
  scriptfile.write("EXEC_DIR=/data/dunham/kallison/paramSearch_2014_11_12\n")
  scriptfile.write("INPUT_FILE=%s\n" %initFile)
  scriptfile.write("cd $PBS_O_WORKDIR\n")
  scriptfile.write("#\n")

  scriptfile.write("mpirun $EXEC_DIR/main $INPUT_FILE\n" )
  scriptfile.close()

  return scriptName


#create input file
def buildInitFile(mu,baseDir):
    origFile = baseDir + "/basin.in"
    initFile = baseDir + "/init_mu_" + str(mu) + ".in"

    sb.call(["cp",origFile,initFile]) #copy the file to be modified

    # open init file and modify values
    text_file = open(initFile, "r")
    whole_thing = text_file.read()

    # modify prefix on data output
    whole_thing = whole_thing.replace("outputDir = data/", "outputDir = %s/data/mu_%s_" %(baseDir,mu))


    # determine size of problem to run
    Dc = 8.0e-3
    sigmaN = 50.0
    b = 0.02
    Lz = 24.0
    Lb = mu*Dc/sigmaN/b
    Nstar = Lz*5/Lb
    N = math.ceil(Nstar)

    # modify problem size
    whole_thing = whole_thing.replace("Ny = 801", "Ny = %i" %(N+10))
    whole_thing = whole_thing.replace("Nz = 801", "Nz = %i" %(N+10))


    # modify shear modulus (mu)
    whole_thing = whole_thing.replace("muIn = 36", "muIn = " + str(mu))
    text_file.close()


    text_file = open(initFile,"w") # write changes to init file
    text_file.write(whole_thing);
    text_file.close();

    return initFile



# keep log of parameter space search performed
results = open("paramSearchLog.txt",'w')
results.write('--- STARTING SS Parameter Space Test in mu ---\n')
results.write('  basin Depth is 4 km\n')
results.write('\n\n\n')
results.close()


for mu in [36, 18, 12, 9]:
  print "mu = " + str(mu)

  baseDir = "/data/dunham/kallison/paramSearch_2014_11_12"

  initFile = buildInitFile(mu,baseDir)
  scriptName = buildLaunchScript(mu,initFile,baseDir)
  print scriptName
  jobname = "mu_" + str(mu)



  # record job submission in log
  results = open("paramSearchLog.txt",'a')
  results.write("Submitting job: %s\n" %initFile)
  results.write( '   submitting at time: %s\n' %(datetime.datetime.now()) )
  results.close()

  # submit job to queue
  print "Submitting job to queue."

  stat,out=commands.getstatusoutput("qsub %s"%scriptName)
  finished = 0
  interval = 120
  user=os.environ['USER']
  while not finished:
      time.sleep(interval)
      #~#finished,out=commands.getstatusoutput("qstat | grep  kallison | grep -v C")
      finished,out=commands.getstatusoutput("qstat -u kallison | grep %s | grep -v C" %jobname)
      print out

  results = open("paramSearchLog.txt",'a')
  results.write( '   job exited at time: %s\n' %(datetime.datetime.now()) )
  results.write( '\n' )
  results.write( '----------------------------------------------------\n' )
  results.close()


results = open("paramSearchLog.txt",'a')
results.write('\n\n--- FINISHED SS Parameter Space Test in mu ---\n')
results.close()


