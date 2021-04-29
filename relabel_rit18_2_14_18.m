
load('rit18_data.mat');

%% Relabel Training labels
relabeled_training = relabel(train_labels);

%change the name of the first argument to the desired file name.
save('train_labels_', 'relabeled_training'); 
%print("train label")

%% Relabel Validation Label
relabeled_val = relabel(val_labels);

%change the name of the first argument to the desired file name.
save('val_labels', 'relabeled_val');
%print("validation label")

%% Relabel Function
function relabeled = relabel(labels)
relabeled = labels;
[x,y] = size(labels);
labeled_total = 0;
    for i = 1 : x   %%rows
        for j = 1 : y  %%columns
            switch relabeled(i,j) 
                case 1                              %Road Markings
                    relabeled(i,j) = 0;
                case 2                              %Tree
                    relabeled(i,j) = 1;
                    labeled_total = labeled_total+1;
                case 3                              %Building
                    relabeled(i,j) = 0 ;   
                case 4                              %Vehicle
                    relabeled(i,j) = 0;
                case 5                              %Person
                    relabeled(i,j) = 0;
                case 6                              %Lifeguard Chair
                    relabeled(i,j) = 0;
                case 7                              %Picnic Table
                    relabeled(i,j) = 0;
                case 8                              %Black Wood Panel
                    relabeled(i,j) = 0;
                case 9                              %White Wood Panel
                    relabeled(i,j) = 0;
                case 10                             %Orange Landing Pad
                    relabeled(i,j) = 0;
                case 11                             %Buoy
                    relabeled(i,j) = 0;
                case 12                             %Rocks
                    relabeled(i,j) = 0;
                case 13                             %Low-Level Vegetation
                    relabeled(i,j) = 0;
                case 14                             %Grass/Lawn
                    relabeled(i,j) = 2;
                    labeled_total = labeled_total+1;
                case 15                             %Sand/Beach
                    relabeled(i,j) = 0;
                case 16                             %Water (lake)
                    relabeled(i,j) = 0;
                case 17                             %Water (pond)
                    relabeled(i,j) = 0;
                case 18                             %Asphalt
                    relabeled(i,j) = 0;
            end
        end
    end
    disp(labeled_total);
    return 
end
