function isPresent = isDatasetPresent(allDatasetPaths,targetDataset)
% checks if dataset is present within hdf5 file

% first check if dataset exists
isPresent = any(ismember(allDatasetPaths,targetDataset));
