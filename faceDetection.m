%Digital forensics 19/20
%Software for face recognition for videos
%auth: Giovanni Gallinaro
%year: 2020

clear
close all

%load library for helpful functions
addpath('func');

startTime = 76;     % time to start reading the video frames (in seconds)
endTime = 85;        % time to stop reading the video frames

labels = ["Adam Sandler", "Alyssa Milano", "Bruce Willis", "Denise Richards", "George Clooney", "Gwyneth Paltrow"];

v = VideoReader('video/Sandler.mp4');   % read the video file

%% TEST CNN

% CNNNet.mat for stardard CNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AlexNetRetrained.mat for AlexNet %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
newnet = load("CNNNet.mat");
newnet = newnet.newnet;

%% FACE DETECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CHOOSE WHETER TO SHOW AND STORE THE RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% N.B. outputs will be stores just if also show is true %%%%%%%%%%%%%%%%%%%
show = true;
store = false;
target_label_index = 1;     % Adam Sandler as target face

% SET THE UPDATING RATE lr %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lr = 1.01;         % set the rate

tic         % measure the computational time

% CHOOSE THE DETECTION FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % "C" = cropface        "D" = detectFaceParts %
% run the algorithm
[predict, scores, weights, j] = videoFaceDetection(v, startTime, ...
endTime, labels, target_label_index, newnet, lr, show, store, "C");

toc

%% RESULTS

%measure the accuracy
pred = predict(1:j)==labels(target_label_index);
s = size(pred);
acc = sum(pred)/s(1);
fprintf('The accuracy of the test set is %f %% \n', acc * 100);

% plot the heatmap of the face
figure(2)
v.CurrentTime = startTime;
ff = readFrame(v);
imshow(ff);
hold on;
imagesc(weights,  'AlphaData', .5);
hold off;

%% SHOW VIDEO W/ DETECTION

if(store == true)
    %load frames
    imageNames = imageDatastore(strcat('facedetection/', labels(target_label_index)),'IncludeSubfolders',true,'LabelSource','foldernames');
    imageNames = {imageNames.Files};
    imageNames = imageNames{1};
    outputVideo = VideoWriter(strcat('facedetection/Adam Sandler','/','facedetection.avi'));
    outputVideo.FrameRate = v.FrameRate;
    open(outputVideo)
    for i = 1:length(imageNames)
        dir = imageNames(i);
        img = imread(dir{1});
        writeVideo(outputVideo,img)
    end
    close(outputVideo)
end






