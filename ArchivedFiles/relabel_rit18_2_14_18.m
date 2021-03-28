pixel_labels = train_labels;
[x,y] = size(train_labels);
labeld_total = 0;
tic
for i = 1 : x
    for j = 1 : y
        switch pixel_labels(i,j) 
            case 1
                pixel_labels(i,j) = 0;
            case 2
                labeld_total = labeld_total+1;
            case 3
                pixel_labels(i,j) = 0 ;   
            case 4
                pixel_labels(i,j) = 0;
            case 5
                pixel_labels(i,j) = 0;
            case 6
                pixel_labels(i,j) = 0;
            case 7
                pixel_labels(i,j) = 0;
            case 8
                pixel_labels(i,j) = 0;
            case 9
                pixel_labels(i,j) = 0;
            case 10
                pixel_labels(i,j) = 0;
            case 11
                pixel_labels(i,j) = 0;
            case 12
                pixel_labels(i,j) = 0;
            case 13
                pixel_labels(i,j) = 0;
            case 14
                labeld_total = labeld_total+1;
            case 15
                pixel_labels(i,j) = 0;
            case 16
                pixel_labels(i,j) = 0;
            case 17
                pixel_labels(i,j) = 0;
            case 18
                labeld_total = labeld_total+1;
        end
    end
end
toc
save('rit18_asphalt_vegetation', 'pixel_labels');