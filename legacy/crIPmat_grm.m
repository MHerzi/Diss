function out=crIPmat_grm(data)
% PURPOSE:
% Transform a vector into a square matrix row-wise (double and cell)
% INPUTS:
% data:  A kx1 vector to be transformed to a  N-1xN-1 matrix.
%        N must ba solution to the equation k^1-k-2*k=0
%        modification: vector can have either double or cell format
% OUTPUTS:
% out:  a N-1 by N-1 matrix. the elements above the main antidiagonal are
%       filled by the elements of the vector data and all other elements are
%       equal to zeros. For example, if data is: data=[2 3 4 5 6 7]' then
%       the rows of out are given below
% out(:,1)=[2 3 4]
% out(:,2)=[5 6 0]
% out(:,3)=[7 0 0]

% Author: Manthos Vogiatzoglou, Martin Grziska
% last modification: Februar, 2nd, 2010

cols=size(data,2);
mat=ivecl_grm(data(:,1));
[R, C]=size(mat);
if iscell(data) == 1
    if cols==1
        out = cell(C-1);
        for i=1:C-1
            helper = diag(mat,i);
            k=1;
            for j = 1:size(helper,1)
                out{j,i} = helper{k};
                k=k+1;
            end
        end
    else
        out=cell(C-1);
        dum=0;
        for i=1:C-1
            for j=1:C-i
                dum=dum+1;
                out{i,j}=data(dum,:);
            end
        end
    end
else
    if cols==1
        out=ones(C-1);
        for i=1:C-1
            out(:,i)=[diag(mat,i); zeros(i-1,1)];
        end
    else
        out=cell(C-1);
        dum=0;
        for i=1:C-1
            for j=1:C-i
                dum=dum+1;
                out{i,j}=data(dum,:);
            end
        end
    end
end




