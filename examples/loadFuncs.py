import h5py
import numpy as np
import os.path

def loadContext(filePath, keys='default'):
  fileName = '%s_data_context.h5' %(filePath)
  if keys == 'default':
    dom_keys = ['/domain/y','/domain/z']
    fault_keys = ['/fault/a','/fault/b','/fault/Dc','/fault/sNEff','/fault_qd/eta_rad','/fault/cs','/fault_qd/rho','/fault/locked','/fault/prestress']
    mom_keys = ['/momBal/mu','/momBal/cs','/momBal/T',
      '/momBal/diffusionCreep/A','/momBal/diffusionCreep/QR','/momBal/diffusionCreep/m','/momBal/diffusionCreep/n',
      '/momBal/dislocationCreep/A','/momBal/dislocationCreep/QR','/momBal/dislocationCreep/n']
    he_keys = ['/heatEquation/k','/heatEquation/rho','/heatEquation/Qrad','/heatEquation/c','/heatEquation/Tamb','/heatEquation/T','/heatEquation/Gw','/heatEquation/w']
    grainSize_keys = ['/grainSizeEv/wattmeter/A','/grainSizeEv/wattmeter/QR','/grainSizeEv/wattmeter/n','/grainSizeEv/wattmeter/p','/grainSizeEv/wattmeter/f','/grainSizeEv/wattmeter/gamma']
    keys = dom_keys + fault_keys + mom_keys + he_keys + grainSize_keys

  d = {}
  d2 = {}
  txtFileName = "%s_domain.txt" %(filePath)
  if os.path.isfile(txtFileName):
    d2 = load_txt(txtFileName);
    appendDict(d,'domain',d2);

  Ny = d['domain']['Ny']
  Nz = d['domain']['Nz']

  with h5py.File(fileName,'r') as currFile:
    for key in keys:
      vec = loadDataSet(currFile,key)
      if len(vec) != 0:
        vec = reshapeDataSet(vec,Ny,Nz);
        createSubDicts(d,key,vec);

  # also load context variables in txt files
  txtFileName = "%s_mediator.txt" %(filePath)
  if os.path.isfile(txtFileName):
    d2 = load_txt(txtFileName);
    appendDict(d,'med',d2);

  txtFileName = "%s_domain.txt" %(filePath)
  if os.path.isfile(txtFileName):
    d2 = load_txt(txtFileName);
    appendDict(d,'domain',d2);

  txtFileName = "%s_heatEquation.txt" %(filePath)
  if os.path.isfile(txtFileName):
    d2 = load_txt(txtFileName);
    appendDict(d,'heatEquation',d2);

  txtFileName = "%s_fault.txt" %(filePath)
  if os.path.isfile(txtFileName):
    d2 = load_txt(txtFileName);
    appendDict(d,'fault',d2);

  txtFileName = "%s_momBal.txt" %(filePath)
  if os.path.isfile(txtFileName):
    d2 = load_txt(txtFileName);
    appendDict(d,'momBal',d2);

  return d

def load_1D(filePath, indict={}, keys='default', Nz=-1, Ny=-1):
  fileName = '%s_data_1D.h5' %(filePath)
  if keys == 'default':
    time_keys = ['/time/time1D','/time/dt1D']
    fault_keys = ['/fault/slip','/fault/slipVel','/fault/psi','/fault/tauP','/fault/strength','/fault/tauQSP','/fault/Vw','/fault/Tw','/fault/T']
    mom_keys = ['/momBal/bcR','/momBal/bcT','/momBal/bcL','/momBal/bcB','/momBal/bcRShift']
    he_keys = ['/heatEquation/bcR','/heatEquation/bcT','/heatEquation/bcL','/heatEquation/bcB']
    keys = time_keys + fault_keys + mom_keys + he_keys

  with h5py.File(fileName,'r') as currFile:
    for key in keys:
      vec = loadDataSet(currFile,key)

      if len(vec) != 0:
        vec = reshapeDataSet(vec,Ny,Nz);
        createSubDicts(indict,key,vec);

  return indict

def load_2D(filePath, indict={}, keys='default', Nz=-1, Ny=-1):
  fileName = '%s_data_2D.h5' %(filePath)
  if keys == 'default':
    time_keys = ['/time/time2D','/time/dt2D']
    mom_keys = ['/momBal/effVisc','/momBal/sxy','/momBal/sxz','/momBal/gTxy','/momBal/gTxz','/momBal/gVxy','/momBal/gVxz','/momBal/dgVxy','/momBal/dgVxz']
    he_keys = ['/heatEquation/T','/heatEquation/dT','/heatEquation/Q','/heatEquation/Qfric','/heatEquation/Qvisc','/heatEquation/kTz']
    grainSize_keys = ['/grainSizeEv/d','/grainSizeEv/d_t']
    keys = time_keys + mom_keys + he_keys + grainSize_keys


  with h5py.File(fileName,'r') as currFile:
    for key in keys:
      vec = loadDataSet(currFile,key)
      if len(vec) != 0:
        vec = reshapeDataSet(vec,Ny,Nz);
        createSubDicts(indict,key,vec);

  return indict

def load_SS(filePath, keys='default'):
  fileName = '%s_data_steadyState.h5' %(filePath)
  if keys == 'default':
    index_keys = ['/steadyState/SS_index']
    fault_keys = ['/fault/slip','/fault/slipVelocity','/fault/psi','/fault/tauP','/fault/strength','/fault/tauQSP','/fault/Vw','/fault/Tw','/fault/T']
    mom_keys = ['/momBal/effVisc','/momBal/sxy','/momBal/sxz','/momBal/gTxy','/momBal/gTxz','/momBal/gVxy','/momBal/gVxz','/momBal/dgVxy','/momBal/dgVxz']
    he_keys = ['/heatEquation/T','/heatEquation/dT','/heatEquation/Q','/heatEquation/Qfric','/heatEquation/Qvisc','/heatEquation/kTz']
    grainSize_keys = ['/grainSizeEv/d','/grainSizeEv/d_t']
    keys = index_keys + fault_keys + mom_keys + he_keys + grainSize_keys

  d = {}
  d2 = {}
  with h5py.File(fileName,'r') as currFile:
    for key in keys:
      vec = loadDataSet(currFile,key)
      if len(vec) != 0:
        createSubDicts(d,key,vec)

  return d



def load_txt(txtFileName):
  d = {}
  with open(txtFileName,'r') as currFile:
    for currLine in currFile.readlines():
      currStrList = currLine.split(' = ')
      if len(currStrList) == 2:
        key = currStrList[0]

        # remove '#' and any following string characters
        val = currStrList[1].split("#")[0]
        val = currStrList[1].split("\n")[0]

        # check if can convert from str to int or float, then do so
        str2Value(d,key,val)
  return d

# creates nested dictionaries corresponding to parts of key
def createSubDicts(d,key,vec):
  currd = d
  keylist = key.split('/')
  for subkey in keylist[:-1]:
    if subkey != '':
      if subkey not in currd.keys():
        currd[subkey] = {}
      currd = currd[subkey]
  currd[keylist[-1]] = vec

# append dictionary to existing dictionary
def appendDict(d,key,d2):
  if key in d.keys():
    d[key] = dict(d[key], **d2)
  else:
    d[key] = d2;


def loadDataSet(currFile,key):
  try:
    dataset = np.array(currFile[key])
    return dataset
  except KeyError:
    pass

  return np.array([])

def reshapeDataSet(vec,Ny,Nz):
  out = vec

  # check that vec can be reshaped
  if not (Nz > -1 and Ny > -1 and isinstance(vec,np.ndarray)):
    return out

  # otherwise reshape vec
  Nt = vec.shape[0];
  if vec.size == Ny*Nz:
    out = (vec.reshape(Ny,Nz)).squeeze();
  if vec.size == Nt*Ny*Nz:
    out = np.moveaxis(vec.reshape(Nt,Ny,Nz),0,-1).squeeze();
  if vec.size == Nt*Ny:
    out = np.moveaxis(vec.reshape(Nt,Ny),0,-1).squeeze();
  if vec.size == Nt*Nz and Nz != 1:
    out = np.moveaxis(vec.reshape(Nt,Nz),0,-1).squeeze();
  if vec.size == Nt*Nz and Nz == 1:
    out = vec.squeeze();
  return out

def str2FloatList(insStr):
  # convert string containing a list of floats into an array
  if '[' in insStr and ']' in insStr:
    temp = insStr.rstrip('\n').lstrip('[').lstrip(' ').rstrip(']').rstrip(' ')
    strList = temp.split(' ')

    try:
      float(strList[0])
    except:
      return False

    finalList = [float(x) for x in strList]
    return finalList
  else:
    return False

def str2Value(d,key,val):
  temp = str2FloatList(val)
  if not isinstance(temp,bool):
    d[key] = temp
  else:
    try:
      d[key] = int(val)
    except ValueError:
      try:
        d[key] = float(val)
      except ValueError:
        d[key] = val



