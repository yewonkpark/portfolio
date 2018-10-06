function store_result = Corr_Calculation(images_test, images_train, store_result)
%feed the function input (test image, train image, matrix for store the result)

size_test = size(images_test,1);
size_train = size(images_train,1);

for i = 1: size_test
    for j = 1: size_train
        a = images_test{i};
        b = images_train{j};
        corr = sum(sum(a.*b))/sqrt(sum(sum(a.*a))*sum(sum(b.*b))); %calculate correlation
        store_result(i,j) = corr;
    end
end

end