function dataStruct = load_struct(fileName, delim)
% Loads the contents of the file (input: fileName) into a struct (output: dataStruct).
% 
% The file is assumed to consist of 2 columns separated by the specified delimiter.
% For example, if delim = ' = ', then the file is assumed to have the structure:
%
% var1 = 1.0
% var2 = 2.0
%      .
%      .
%      .
%
% The data struct will contain fields with the names {var1,var2,...} with
% the values specified.

    fid = fopen(fileName);

    fileLine = fgetl(fid);
    while ischar(fileLine)
          matches = strfind(fileLine,delim);
          fieldName = fileLine(1:matches);
          fieldValue = fileLine(matches+3:end);
          
          fieldName = genvarname(fieldName);
          
          dataStruct.(fieldName) = str2num(fieldValue);
          if isempty(dataStruct.(fieldName)),
            dataStruct.(fieldName)=fieldValue;
          end
          fileLine = fgetl(fid); % drops end-of-line character
    end
    fclose(fid);

end