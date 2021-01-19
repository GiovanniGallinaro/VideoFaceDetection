%Digital forensics 19/20
%Reference software for face recognition
%prepare dataset with masks
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

%load the face detector
faceDetector = vision.CascadeObjectDetector;
%load the mouth detector
mouthDetector = vision.CascadeObjectDetector('Mouth','MergeThreshold',8);
%load the nose detector
noseDetector = vision.CascadeObjectDetector('Nose','MergeThreshold',8);

%load an image augmenter setting 
%random rotations [-10, 10]°,
%random scaling   [ 0.9, 1.1],
%random translation [-10,10] x [-10,10] pixels
imageAugmenter = imageDataAugmenter('RandRotation',[-10 10 ],...
    'RandScale',[0.9 1.1],'RandXTranslation',[-10 10],...
    'RandYTranslation',[-10 10]);

%% Load Image Information from face database directory (each folder = 1 person)
faceDatabase = imageSet(str_arc,'recursive');

%Npeople is the number of subjects
Npeople = 3;

%create folders of masked subjects: croppedFacesMask
%first remove existing folders
[s , m , mid] = rmdir('croppedfacesMask','s');

%loop on each subject/person
for i =1:Npeople
    %create dir
    [s , m , mid] = mkdir([ 'croppedfacesMask' filesep faceDatabase(i).Description]);
    str = [str_arc filesep faceDatabase(i).Description];
    %load the dataset
    ds1 = imageDatastore(str,'IncludeSubfolders',true,'LabelSource','foldernames');
    
    %count the number of images for the current subjects (training set)
    T = countEachLabel(ds1);
    Nfaces = T(1,2).Variables; %number of faces for the current person
    
    j=1;
    for f = 1:Nfaces
        i1orig = readimage(ds1,f); %read the current image for person i
        
        i1=i1orig;
        
        for r=1:1  %number or realizations
            %find the face
            bboxF=step(faceDetector,i1);
            if isempty(bboxF)
                continue;
            end
            
            %create covering mask for mouth
            limit_y0=round(bboxF(1,1) + .5*bboxF(1,3));
            limit_y1=round(bboxF(1,1) + .95*bboxF(1,3));
            limit_x0=round(bboxF(1,1)+.15*bboxF(1,3));
            limit_x1=round(bboxF(1,1)+.85*bboxF(1,3));
            i1(limit_y0:limit_y1,limit_x0:limit_x1,:)=0;
            
            [img,face] = cropface(i1); %select the face inside the current image
            if face==1 %face is detected
                %write image
                imwrite(img,['croppedfacesMask', filesep,  faceDatabase(i).Description, filesep, int2str(j), '.jpg']);
                j = j+1;
            end
            %augment data: apply a random affine transformation
            i1 = augment(imageAugmenter,i1orig);
        end
    end
end

%%Verify efficiency of the trained networks
%load test set
 Test = imageDatastore('croppedfacesMask','IncludeSubfolders',true,'LabelSource','foldernames');
 % Resize the images to the input size of the net
 Test.ReadFcn = @(loc)imresize(imread(loc),[64,64]);  %CNN wants 64x64
 
 %load the trained network
 load CNNnet;
 %classify input image
 [predict,scores] = classify(newnet,Test);
 %verify accuracy of the classification (number of classes corresponding
 %to the ground truth
 names = Test.Labels;
 pred = (predict==names);
 s = size(pred);
 acc = sum(pred)/s(1); %accuracy
 fprintf('The accuracy of the test set is %f %% \n',acc*100);
 %plot confusion matrices
 plotconfusion(names, predict);
 