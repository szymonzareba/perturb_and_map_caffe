function image = visualiseParameters(F, xSize, ySize)

fullsize = size(F,1);

numberOfImages = size(F,2);
gridSize = round(sqrt(numberOfImages))+1;

image = [];

xPos = 1;
yPos = 1;
for no=1:numberOfImages
    if xPos == 1
        if yPos == 2
            image = [wiersz];
        else
            if yPos > 2
                image = [image ; zeros(2, size(wiersz,2)) ; wiersz];
            end
        end
        wiersz = [reshape(F(:,no),xSize,ySize)];

    else
        wiersz = [wiersz zeros(xSize, 2) reshape(F(:,no),xSize,ySize)];
    end
    xPos = xPos + 1;
    if xPos == gridSize
        xPos = 1;
        yPos = yPos + 1;
    end
end
image = [image ; zeros(2, size(image,2)) ; [wiersz zeros(size(wiersz,1),size(image,2)-size(wiersz,2))]];
end