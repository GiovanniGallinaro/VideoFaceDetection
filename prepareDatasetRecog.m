%Digital forensics 19/20
%Reference software for face recognition
%create a simple CNN for face recognition
%auth: Giovanni Gallinaro
%year: 2020

clear
close all

%load library for face alignment
addpath('func');

%load an image augmenter setting 
%random rotations [-10, 10]°,
%random scaling   [ 0.9, 1.1],
%random translation [-10,10] x [-10,10] pixels
imageAugmenter = imageDataAugmenter('RandRotation',[-10 10 ],...
    'RandScale',[0.9 1.1],'RandXTranslation',[-10 10],...
    'RandYTranslation',[-10 10]);

% %load data
% im = imageDatastore('croppedfaces','IncludeSubfolders',true,'LabelSource','foldernames');
% % Resize the images to the input size of the net
% im.ReadFcn = @(loc)imresize(imread(loc),[64,64]);
% %%% imgs = readall(imds); % to read all images
% %%% img = readimage(imds,i); % read index i

startTime = 76;     % time to start reading the video frames (in seconds)
endTime = 85;        % time to stop reading the video frames

labels = ["Adam Sandler", "Alyssa Milano", "Bruce Willis"];

v = VideoReader('video/Sandler.mp4');   % read the video file

v.CurrentTime = startTime;
frame = readFrame(v);       % read the frame
frame_size = size(frame);
frame_size = frame_size(1:2);

%% STORE VIDEO FRAMES

[s , m , mid] = rmdir('videoframes/', 's');  % remove existing dir

j = 1;
v.CurrentTime = startTime;
while (v.CurrentTime < endTime)
    frame = readFrame(v);       % read the frame
    [s , m , mid] = mkdir(strcat('videoframes/', labels(1)));   % create dir
    imwrite(frame, strcat('videoframes/', labels(1), "/", int2str(j), '.jpg'));       %write image
    j = j+1;
end

%% CROP FRAMES

[s , m , mid] = rmdir('croppedvideoframes/', 's');  % remove existing dir

j = 1;
v.CurrentTime = startTime;
while (v.CurrentTime < endTime)
    frameorig = readFrame(v);       % read the frame
    [s , m , mid] = mkdir(strcat('croppedvideoframes/', labels(1)));   % create dir
    
    frame = frameorig;
            
    [img, face] = cropface(frame);  %select the face inside the current image
    if face == 1  %face is detected
        imwrite(img,strcat('croppedvideoframes/', labels(1), int2str(j), '.jpg'));       %write image
        j = j+1;
    end
end





