function image = normalizeParameters(F)
numberOfImages = size(F,2);
image = [];
    for no=1:numberOfImages
        image = [image (F(:,no)-min(F(:,no)))/max(F(:,no)-min(F(:,no)))];
    end
end