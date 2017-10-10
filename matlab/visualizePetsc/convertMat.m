function convertMat(A)
% clear all

% fileID = fopen('test_output.txt','w');

% [D Pinv D2 S Q] = secondOrderSBPoperators(2,8);
% A = S;%sparse([0 0 0;0 1 2; 0 1 3]);
[rowVec,colVec,vals] = find(A);
mat = sortrows([rowVec,colVec,vals]); allones = ones(length(rowVec),1);
rowVec = mat(:,1)-allones; colVec = mat(:,2)-allones; vals = mat(:,3);


ind = 1; currRow=0;
mystring = '';
while currRow < size(A,1)
    temp = sprintf('row %u:',currRow);
    mystring = strcat(mystring,temp);
    while ind < length(rowVec)+1 && rowVec(ind)==currRow
        if mystring(end)==':'
            temp = sprintf(' (%u, %g)',colVec(ind),vals(ind));
        else
            temp = sprintf('  (%u, %g)',colVec(ind),vals(ind));
        end
        mystring=strcat(mystring,temp);
        ind = ind+1;
    end
    currRow = currRow+1;
    mystring = strcat(mystring,' \n');
end
%mystring = strcat(mystring,'\n');


% fprintf(fileID,mystring);
sprintf(mystring)

% fclose(fileID);
end
