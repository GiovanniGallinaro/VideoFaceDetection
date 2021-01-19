%Digital forensics 19/20
%Reference software for face recognition
%create a simple CNN for face recognition
%auth: Simone Milani
%year: 2020

clear
close all

%load library for face alignment
addpath('func');

%Npeople is the number of subjects
Npeople = 6;

%load training data
im = imageDatastore('croppedfaces','IncludeSubfolders',true,'LabelSource','foldernames');
% Resize the images to the input size of the net
im.ReadFcn = @(loc)imresize(imread(loc),[64,64]);
%Split the training set into training (80%) and validation (20%) 
[Train ,Validation] = splitEachLabel(im,0.8,'randomized');
%Load test data
Test = imageDatastore('croppedfacesTest','IncludeSubfolders',true,'LabelSource','foldernames');
% Resize the images to the input size of the net
Test.ReadFcn = @(loc)imresize(imread(loc),[64,64]);
 
 
%Network structure
%Sequence of layers for the network
layer_vet=[
    imageInputLayer([64 64 3])
     
    convolution2dLayer([8 8],8)
    batchNormalizationLayer
    reluLayer();
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer([8 8],32);
    batchNormalizationLayer
    reluLayer();
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer([8 8],64);
    batchNormalizationLayer
    reluLayer();
    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(Npeople)

    softmaxLayer();

    classificationLayer()
 ];
 

 % options for training the net if your newnet performance is low decrease
 % the learning_rate
 learning_rate = 0.00003;
 opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,...
     'MaxEpochs',5,...
     'MiniBatchSize',64,...
     'ValidationData',Validation,...
     'Plots','training-progress');
 
 %training networks
 [newnet,info] = trainNetwork(Train, layer_vet, opts);
 
 %predict the labels for the test set
 [predict,scores] = classify(newnet,Test);
 
 %measure the accuracy
 names = Test.Labels;
 pred = (predict==names);
 s = size(pred);
 acc = sum(pred)/s(1);
 fprintf('The accuracy of the test set is %f %% \n', acc * 100);
 
 nntraintool close
 %plot confusion matrix
 plotconfusion(names, predict);
 
 %show the computed filters for layer 2
 figure(2);
 I2=deepDreamImage(newnet,2,1:3);
 montage(I2);
 
  %show the computed filters for layer 6
  figure(3);
 I6=deepDreamImage(newnet,6,1:8);
 montage(I6);
 
  %show the computed filters for layer 10
  figure(4);
 I10=deepDreamImage(newnet,10,1:32);
 montage(I10);
 
 %save the network into CNNNet.mat
  save CNNNet newnet
 