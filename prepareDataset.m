%Digital forensics 19/20
%Reference software for face recognition
%auth: Simone Milani
%year: 2020

clear
close all

%load library for face alignment
addpath('func');

%load dataset
if ispc
    str_arc='data\PeopleFace';
else
    str_arc='data/PeopleFace';
end

%load the face detector (not used)
faceDetector = vision.CascadeObjectDetector;

%load an image augmenter setting 
%random rotations [-10, 10]°,
%random scaling   [ 0.9, 1.1],
%random translation [-10,10] x [-10,10] pixels
imageAugmenter = imageDataAugmenter('RandRotation',[-10 10 ],...
    'RandScale',[0.9 1.1],'RandXTranslation',[-10 10],...
    'RandYTranslation',[-10 10]);


%% Load Image Information from face database directory (each folder = 1 person)
faceDatabase = imageSet(str_arc,'recursive');

% n is the number of subjects
Npeople = 6;

%create folders for the training data/test data
%trainig data: folder 'croppedfaces'
%test data: folder 'croppedfacesTest'
%first remove existing folders
[s , m , mid] = rmdir('croppedfaces','s');
[s , m , mid] = rmdir('croppedfacesTest','s');

%loop on each subject/person
for i =1:Npeople
    %create dirs
    [s , m , mid] = mkdir([ 'croppedfaces' filesep faceDatabase(i).Description]);
    [s , m , mid] = mkdir([ 'croppedfacesTest' filesep faceDatabase(i).Description]);
    str = [str_arc filesep faceDatabase(i).Description];
    %load the dataset
    ds1 = imageDatastore(str,'IncludeSubfolders',true,'LabelSource','foldernames');
    
    %separate the dataset in training and test
    [ds1_train ,ds1_test] = splitEachLabel(ds1,0.8,'randomized');
    
    %count the number of images for the current subjects (training set)
    T = countEachLabel(ds1_train);
    Nfaces = T(1,2).Variables;  %number of faces for the current person
    
    j=1;
    for f = 1:Nfaces
       i1orig = readimage(ds1_train,f);  %read the current image for person i
        
       i1=i1orig;
        
        for r=1:10  %number or realizations
        [img,face] = cropface(i1);  %select the face inside the current image
            if face==1  %face is detected
                %write image
                imwrite(img{1, 1},['croppedfaces', filesep,  faceDatabase(i).Description, filesep, int2str(j), '.jpg']);
                j = j+1;
            end
            %augment data: apply a random affine transformation
            i1 = augment(imageAugmenter,i1orig);
        end
    end
    
    %count the number of images for the current subjects (test set)
    T = countEachLabel(ds1_test);
    Nfaces = T(1,2).Variables;  %number of faces for the current person
    
    j=1;
    for f = 1:Nfaces
       i1orig = readimage(ds1_test,f); %read the current image for person i
        
       i1=i1orig;
        
        for r=1:10  %number or realizations
        [img,face] = cropface(i1); %select the face inside the current image
            if face==1  %face is detected
                %write image
                imwrite(img{1, 1},['croppedfacesTest', filesep,  faceDatabase(i).Description, filesep, int2str(j), '.jpg']);
                j = j+1;
            end
            %augment data: apply a random affine transformation
            i1 = augment(imageAugmenter,i1orig);
        end
    end
end

 