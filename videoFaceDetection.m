% videoFaceDetection: detect and classify faces in a video sequence
%
% [predict, scores, weights, j] = ...
% videoFaceDetection(v, startTime, endTime, labels, target_label_index,
% newnet, lr, show, store);
%
%Output parameters:
% predict = predicted labels for each frame (size = # of frames)
% scores = array of classification scores (size = # of frames x # of
% labels)
% weights = matrix containing the weights for each pixels
% j = # of frames
%
%Input parameters:
% v = video file (VideoReader object)
% startTime = time from which the algorithm should start running (s)
% endTime = time in which the algorithm should end its execution (s)
% labels = array of possible labels
% target_label_index = index of the target label (label to be detected)
% newnet = trained neural network
% lr = updating rate for the algorithm
% show = boolean variable. If true, the function shows the output in real
% time
% store = boolean varialbe. If true, the functions stores the output
% func = function to use for the detection: "C" = cropface
%                                           "D" = detectFaceParts
%
%
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Video Face Detection:                                    %
%                   Giovanni Gallinaro                     %
%                   Univ. of Padova                        %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [predict, scores, weights, j] = videoFaceDetection(v, startTime, endTime, labels, t_index, newnet, lr, show, store, func)
    
    if ~exist('show','var')
        show = false;
    end
    
    if ~exist('store','var')
        store = false;
    end

    % store useful variables
    v.CurrentTime = startTime;
    frame = readFrame(v);       % read the frame
    frame_size = size(frame);
    frame_size = frame_size(1:2);   % size of the frames
    n_faces = length(labels);   % number of faces

    detector = buildDetector();

    [s , m , mid] = rmdir('facedetection/', 's');  % remove existing dir
    [s , m , mid] = mkdir(strcat('facedetection/', labels(t_index)));   % create dir

    v.CurrentTime = startTime;
    j = 1;
    n_frames = ceil((endTime - startTime + 1)*v.frameRate);

    % to estimate accuracy
    predict = strings(n_frames, 1);
    scores = zeros(n_frames, n_faces);

    weights = ones(frame_size);     % initialize the weights matrix
    
    xbar=waitbar(0, 'Starting the execution...');

    while (v.CurrentTime < endTime)        % for every frame
        frameorig = readFrame(v);       % read the frame
        frame = frameorig;

        if strcmp(func, "C")
            [faces,~,bbox] = cropface(frame);
        else
            [bbox,bbX,faces,bbfaces] = detectFaceParts(detector, frame);
        end 
        
        if ~isempty(bbox)       % if at least one face is detected
            frame_predictions = strings(length(faces), 1);
            frame_scores = zeros(length(faces), n_faces);

            for i = 1:length(faces)     % for each detected face
                % check weights
                face_box = bbox(i, 1:4);
                w_sum = sum(weights(face_box(2):face_box(2)+face_box(4)-1, face_box(1):face_box(1)+face_box(3)-1), 'all');
                s_w = size(weights(face_box(2):face_box(2)+face_box(4)-1, face_box(1):face_box(1)+face_box(3)-1));
                m_size = s_w(1)*s_w(2);
                r_weight = w_sum/m_size;

                % classification
                inputSize = newnet.Layers(1).InputSize;
                ff = imresize(faces{i},inputSize(1:2));         % resize to fill in the network
                [pred_temp, score_temp] = classify(newnet, ff);     % get the prediction and score on the current detected face

                % update the score accordingly to the weights
                score_temp(t_index) = score_temp(t_index)*r_weight;

                % store date
                [~, temp_ind] = max(score_temp);
                frame_predictions(i, 1) = labels(temp_ind);     % store the predictions
                frame_scores(i, :) = score_temp;            % and the scores
            end

            % store correctly detected image - best match wrt prediction
            indexes = find(frame_predictions(:, 1) == labels(t_index));       % find indexes where predict = test label
            [~, correct_ind] = max(frame_scores(:, 1));         % find maximum score in case
            max_score = frame_scores(correct_ind, :);

            % update the weights
            c_face = bbox(correct_ind, 1:4);            % box of the correctly identified face
            new_weights = (weights(c_face(2):c_face(2)+c_face(4)-1, c_face(1):c_face(1)+c_face(3)-1))*lr;
            weights(c_face(2):c_face(2)+c_face(4)-1, c_face(1):c_face(1)+c_face(3)-1) = new_weights;

            % store prediction and score
            predict(j) = frame_predictions(correct_ind, 1);
            scores(j, :) = max_score;

            if length(indexes) >= 1      % in case of conflicting predictions
                indexes(indexes == correct_ind) = [];          % remove correct index
            end

            if show == true
                % assing rectangle colors
                rectcolor = strings(length(faces), 1);
                rectcolor(:, 1) = char("blue");
                rectcolor(correct_ind, 1) = char("green");
                if ~isempty(indexes)
                    rectcolor(indexes, 1) = char("red");
                end
                rectcolor = cellstr(rectcolor);

                for i = 1:length(faces)     % for each detected face
                    frame = insertText(frame, bbox(i, 1:2), char(frame_predictions(i, 1)), 'AnchorPoint','LeftBottom', 'FontSize', 18);
                    frame = insertShape(frame, 'Rectangle', bbox(i, 1:4), 'LineWidth', 2, 'Color', rectcolor(i));
                end
            end
        end

        if show == true
            imshow(frame);
            drawnow();
            if store == true
                imwrite(frame, strcat('facedetection/', labels(t_index), "/", int2str(j), '.jpg'));       %write image
            end
        end
        %j
        waitbar((j-1)/271, xbar, strcat('In progress (Frame', {' '}, string(j-1),')'));
        j = j+1;
    end
    
    close(xbar)

end