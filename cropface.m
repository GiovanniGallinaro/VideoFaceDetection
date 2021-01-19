function [J,face,box] = cropface(img)
FaceDetect = vision.CascadeObjectDetector('FrontalFaceCART','MinSize',[150,150]);
bbox=step(FaceDetect,img);
if ~isempty(bbox)
    J = cell(size(bbox,1),1);
    for i=1:size(bbox,1)
    J{i,1} = imcrop(img,bbox(i,:));
    end
    face = 1;
    box = bbox;         % return coordinates of the box where the face was detected
else
    J{1,1} = img;
    face = 0;
    box = [0, 0, 0, 0];     % draw nothing if face is not detected
end
end