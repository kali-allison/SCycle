function isPresent = ismember_hdf5(info, targetGroup, targetDataSet)
% check if group and dataset are both present in hdf5 file

% default output is: isPresent = false
isPresent = 0;

% check if group is present
groupNames = recursiveGroupFinder(info.Groups);  % full list of groups in hdf5 file
[groupIsPresent, groupLoc] = ismember(targetGroup,groupNames);
groupIsPresent

% if group is present, check if dataset is present
if groupIsPresent
  dataSets = info.Groups(groupLoc).Datasets;
  dataSetNames = {dataSets.Name};
  isPresent = ismember(targetDataSet,dataSetNames);
end
