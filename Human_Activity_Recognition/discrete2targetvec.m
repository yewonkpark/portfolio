
function targetvec = discrete2targetvec(input,labNum)
sampleNumber = size(input,1);
targetvec = zeros(sampleNumber,labNum); %generate matrix of zeros size of labNum

%replace zero with 1 according to the label for each sample
for n = 1:sampleNumber;
    targetvec(n,input(n)) = 1; 
end

end
