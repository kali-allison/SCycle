function dataStruct = loadStruct(fileName, delim)
% Loads the contents of the file (input: fileName) into a struct (output: dataStruct).
%
% The file is assumed to consist of 2 columns separated by the specified delimiter.
% For example, if delim = ' = ', then the file is assumed to have the structure:
%
% var1 = 1.0
% var2 = 2.0
% var3 = 3.0 # comments are formatted like this
%      .
%      .
%      .
%
% The data struct will contain fields with the names {var1,var2,...} with
% the values specified. Note that text fields are returned as character
% arrays. Also, this IS WHITE SPACE SENSITIVE except at the ends of lines.

fid = fopen(fileName);

while ~feof(fid)  
  fileLine = fgetl(fid); % load current line
  
   % skip empty lines
  if isempty(fileLine) || ~ischar(fileLine), continue, end
  
  % split line at delimiter
  matches = strfind(fileLine,delim);
  fieldName = fileLine(1:matches-1);
  fieldValue = fileLine(matches+3:end);
  
  % remove any trailing comments
  commentIndex = strfind(fieldValue,'#');
  if ~isempty(commentIndex)
    fieldValue = fieldValue(1:commentIndex-1);
  end
  
  fieldName = genvarname(fieldName);
  
  dataStruct.(fieldName) = str2num(fieldValue);
  if isempty(dataStruct.(fieldName)),
    dataStruct.(fieldName)=fieldValue;
  end

end
fclose(fid);

end