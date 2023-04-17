clear;
clc;

%% step1:Display the original image on screen

img = imread('charact2.bmp');
figure()
imshow(img)
title('Original Image');

%% step2:Implement and apply mask

%{
Mean filter:
The third input parameter is the edge fill method, selected as 'zero' or 'replicate'.
However, the mean filter has little impact on this assignment because the objective 
is to obtain the outline of the letters, which is unrelated to the image edges.
Therefore, edge processing is not a concern for the subsequent functions in this task.
%}

%{
img_mean = mean_filter(img, 3, 'replicate'); 
figure()
imshow(img_mean)
title('Image after applying 3x3 averaging mask');
img_mean = mean_filter(img, 7, 'zero');
figure()
imshow(img_mean)
title('Image after applying 7x7 averaging mask'); 
%}
% The larger the mask size, the more blurred the filtering result.

% Manual programming of median filter
%{
img_median = median_filter(img, 3);
figure()
imshow(img_median)
title('Image after applying 3x3 median filter');
%}

img_median = median_filter(img, 7);
figure()
imshow(img_median)
title('Image after applying 7x7 median filter');

% Manual programming of sharpening
%{
img_sharpened = sharpening(img);
figure()
imshow(img_sharpened)
title('Image after applying laplacian filter');
%}
% Sharpening sharpens the character outline but also introduces low frequency noise

% rotating mask
%{
% the three input parameters are:
% the input image, the maximum angle, the rotation step and the mask size.
img_rotated = rotate(img,360,45,3);
figure()
imshow(img_rotated)
title('Image after averaging by using a 3x3 rotation mask')
img_rotated = rotate(img,360,45,7);
figure()
imshow(img_rotated)
title('Image after averaging by using a 7x7 ratation mask')
%}

% image augmentation 
%{
% Combined enhancement: 
% sharpening of the image, followed by median filtering to remove low frequency noise
img_sharpened = sharpening(img);
img_uint8 = uint8(img_sharpened.*255);  % Type conversion
img_aug = median_filter(img_uint8, 7);
figure()
imshow(img_aug)
title('Image after augmentation');
%}
img = img_median;

%% step3:Create a sub-image 

[h, w] = size(img); % Get image size

% Using the slice operation to get only the bottom half of the image
% img_bottom = img(y_start:y_end, x_start:x_end, :);
img_bottom = img(ceil(h/2):h, :, :);    

figure();
imshow(img_bottom);
title('Sub-image');

%% step4:Create a binary image

% img=rgb2gray(img); 
% Programming manually: conversion of colour images to grey scale first
[m,n,l]=size(img_bottom); 
pixel=zeros(m,n);  % Create m*n zero matrix
for i=1:m
    for j=1:n
        pixel(i,j)=0.299*img_bottom(i,j,1)+0.587*img_bottom(i,j,2)+0.114*img_bottom(i,j,3);
    end
end
img_bottom=uint8(pixel); % Generate a new grayscale image

I=img_bottom;

% Set initial threshold: the middle value of the maximum and minimum values
zmax=max(I(:));     
zmin=min(I(:));
T1=(zmax+zmin)/2;
disp(strcat('Initial threshold T1:',num2str(T1)));
[m ,n]=size(I);

% Segment the image into foreground and background according to the threshold value, 
% and find the average grey level z1 and z2 of both respectively
while true
        ifg=0;	% Total number of foreground pixel points
        ibg=0;	% Total number of background pixel points
        fnum=0;	% The sum of the pixel values classified as foreground
        bnum=0;	% The sum of the pixel values classified as background

        for i=1:m
            for j=1:n
                tmp=I(i,j);
                if(tmp>=T1)
                    ifg=ifg+1;
                    fnum=fnum+double(tmp);  
                else
                    ibg=ibg+1;
                    bnum=bnum+double(tmp);
                end
            end
        end

        % Calculate the average of foreground and background
        z1=fnum/ifg;
        z2=bnum/ibg;

        if(T1==(uint8((z1+z2)/2)))
            break;
        else
            T1=uint8((z1+z2)/2);
        end

        % Exit the iteration when the threshold does not transform
end

disp(strcat('Updated threshold T2:',num2str(T1)));

% Binarization process
% img_binary=imbinarize(I,T1/255) 
img_binary = zeros(m,n);
for i=1:m
	for j=1:n
		if(I(i,j)>=double(T1))
			img_binary(i,j)=1;
		end
	end
end

figure()
imshow(img_binary)
title('Binary image');

%% step5:Determine the outline(s) of characters

% Manual segmentation of certain letters
[height, width] = size(img_binary);

for i=1 : floor((height+1))
    img_binary(i,146)=0;
    img_binary(i,240)=0;
    img_binary(i,503)=0;
    img_binary(i,593)=0;
    img_binary(i,857)=0;
end

% Connected area marking
[stats,L, num] = connected_components(img_binary);
fprintf('%d',num)

% Removal of noise point areas
% Find areas with less than 1000 pixels and remove these noisy areas
for i=1:length(stats)
    area=stats(i).Area;
    plist=stats(i).PixelList;
    if area<1200
        for j=1:size(plist,1)
            L(plist(j,1),plist(j,2))=0;
        end
    end
end

[~,L, num] = connected_components(logical(L));
fprintf("%d",num)

% Sobel operator edge detection on grayscale images
gx = [-1 0 1; -2 0 2; -1 0 1];
gy = [-1 -2 -1; 0 0 0; 1 2 1];
% Convolution
img_gx = doconv(double(L), gx);
img_gy = doconv(double(L), gy);
img_edges = sqrt(img_gx.^2 + img_gy.^2);

% Positioning of character outlines according to edge position
img_edges(img_edges < 0.4) = 0;
img_edges(img_edges >= 0.4) = 1;

% Show character outline
figure()
imshow(img_edges)
title('Character Outline');

%% step6:Segment and label

% Converting colour labels to rgb
img_labeled=label2rgb(L);
% Show rgb characters after marker
figure()
imshow(img_labeled);
title('Labeled Characters');

% Splitting characters
stats = regionprops(L, 'BoundingBox');
for i = 1:num
    bbox = stats(i).BoundingBox;
    pos=[bbox(1),bbox(2),bbox(3),bbox(4)];
    % Cutting characters using boundingbox
    img_1=imcrop(L, pos);
    resized_img = imresize(img_1, [128, 128]);
    % Use the imcomplement function to invert the image 
    % so that the black and white colours of the image are reversed
    resized_img=imcomplement(resized_img);
    size(resized_img)
    figure()
    imshow(resized_img);  
end

%% Functions step2:

function output_image = rotate(input_image, max_angle, step_size, window_size)

% Convert input image to grayscale if necessary
if size(input_image,3) > 1
    input_image = rgb2gray(input_image);
end

% Convert input image to double
input_image = im2double(input_image);

% Define output image
output_image = zeros(size(input_image));

% Define maximum allowed angle for rotation
max_angle = max_angle / 2;

% Define step size for rotation
step_size = step_size / 2;

% Define window size for filtering
half_win = floor(window_size / 2);

% Loop through all pixels in input image
for i = 1:size(input_image,1)
    for j = 1:size(input_image,2)
        
        % Define minimum variance and corresponding mask
        min_var = inf;
        min_mask = zeros(window_size);
        
        % Loop through all possible angles of rotation
        for angle = -max_angle:step_size:max_angle
            
            % Define rotation matrix
            R = [cosd(angle) -sind(angle); sind(angle) cosd(angle)];
            
            % Define rotated window
            x_range = max(1, i-half_win):min(size(input_image,1), i+half_win);
            y_range = max(1, j-half_win):min(size(input_image,2), j+half_win);
            window = input_image(x_range, y_range);
            rotated_window = imrotate(window, angle, 'crop');
            
            % Compute variance of rotated window
            var_rotated = var(rotated_window(:));
            
            % Update minimum variance and corresponding mask
            if var_rotated < min_var
                min_var = var_rotated;
                min_mask = rotated_window;
            end
            
        end
        
        % Apply minimum variance mask to current pixel
        output_image(i,j) = mean(min_mask(:));
        
    end
end

% Convert output image to uint8
output_image = im2uint8(output_image);

end

function filtered_img = sharpening(input_image)
% input_image：RGB images

% Image sharpening using the Laplace operator
rgb = im2double(input_image);
r_channel = rgb(:,:,1);              % Three-channel separation of RGB images
g_channel = rgb(:,:,2);              
b_channel = rgb(:,:,3);
L_filter = [1 1 1;               % Laplace operator, 8-adjacent mask
            1 -8 1;
            1 1 1];

% Sharpening of the three channels separately
filtered_red = myConv2(r_channel, L_filter);
filtered_green = myConv2(g_channel, L_filter);
filtered_blue = myConv2(b_channel, L_filter);
% Merging three channels
filtered_img = cat(3, filtered_red, filtered_green, filtered_blue);

% Since the central coefficient of the Laplace template is negative, 
% the negative sign is taken here
sharpened_img = rgb - filtered_img;
% sharpened_img = imadjust(sharpened_img, [0.2 0.8], []);
% Adjusting Contrast

filtered_img = sharpened_img;            % Back to Sharpening images

% Convolutional algorithms
function result = myConv2(image, kernel)
    [m, n] = size(image);
    [p, q] = size(kernel);
    padded_image = padarray(image, [floor(p/2) floor(q/2)]);
    result = zeros(size(image));
    for i = 1:m
        for j = 1:n
            temp = padded_image(i:i+p-1, j:j+q-1) .* kernel;
            result(i, j) = sum(temp(:));
        end
    end
end

end

function filtered_image = mean_filter(input_image, filter_size, padding_type)
% input_image: RGB input image
% filter_size: Filter size (assumed to be square, e.g. 3x3, 5x5, etc.)
% padding_type: Edge processing method with the value 'zero' or 'replicate'

    [m, n, ~] = size(input_image);
    filtered_image = zeros(m, n, 3, 'uint8'); % Modify data type

    % Radius of the filter
    r = floor(filter_size / 2);

    % Edge treatment
    if strcmp(padding_type, 'zero')
        % Complementary zeros at the edge
        input_image = padarray(input_image, [r, r], 0);
    elseif strcmp(padding_type, 'replicate')
        % Copy pixels at the edge
        input_image = padarray(input_image, [r, r], 'replicate');
    else
        error('Ineffective edge treatment');
    end

    % Loop through each pixel
    for i = 1:m
        for j = 1:n
            % Extracts a neighbourhood of the corresponding size, centred on the current pixel
            x_min = i;
            x_max = i + filter_size - 1;
            y_min = j;
            y_max = j + filter_size - 1;
            neighborhood = input_image(x_min:x_max, y_min:y_max, :);
            
            % Average the pixels in the neighbourhood as the output value of the current pixel
            filtered_image(i, j, :) = uint8(mean(mean(neighborhood))); % Modify data type
        end
    end
end

function filtered_image = median_filter(input_image, filter_size)
% image: RGB image
% filter_size: the mask size

% Split RGB image into 3 channels
r_channel = input_image(:,:,1);
g_channel = input_image(:,:,2);
b_channel = input_image(:,:,3);

% Calculate the size of image
[rows, cols] = size(r_channel);

% Initialize output image
filtered_image = zeros(rows, cols, 3, 'uint8');

% Apply median filter to each channel
for i = 1:3
    % Extract the current channel
    channel = input_image(:,:,i);
    
    % Median filter
    filtered_channel = zeros(rows, cols, 'uint8');
    for x = 1:rows
        for y = 1:cols
            % Calculate the window
            row_range = max(1, x - floor(filter_size/2)) : min(rows, x + floor(filter_size/2));
            col_range = max(1, y - floor(filter_size/2)) : min(cols, y + floor(filter_size/2));
            
            % Extract the pixels in current window
            window_pixels = channel(row_range, col_range);
            
            % Sort the pixels in the window and find the median value
            filtered_channel(x, y) = median(window_pixels(:));
        end
    end
    
    % Merge the channels
    filtered_image(:,:,i) = filtered_channel;
end

% Convert output image to RGB image
filtered_image = uint8(filtered_image);
end

%% Functions step5 and step6:
function [stats,L, num] = connected_components(BW)
    [height, width] = size(BW);
    L = zeros(height, width);
    num = 0;
    
    % Define the position offset of adjacent pixels
    offsets = [-1, 0; 1, 0; 0, -1; 0, 1];
    
    % Iterate over all pixels in the image
    for i = 1:height
        for j = 1:width
            if BW(i, j) == 1 && L(i, j) == 0 % is a foreground pixel and is not accessed
                num = num + 1;
                queue = [i, j];
                L(i, j) = num;  
                queue_store=[];
                % Connected area marking using queue scanning algorithm
                while ~isempty(queue)
                    % Fetch the first pixel in the queue and get its adjacent pixels
                    p = queue(1, :);
                    queue(1, :) = [];
                    queue_store=[queue_store;p];
                    % @plus means + and can be written as neighbors = p+offsets.
                    neighbors = bsxfun(@plus, p, offsets);
                    
                    % Check that the neighbouring pixels are foreground pixels and are not marked
                    % If yes, mark it as the current connected area and add it to the queue
                    for k = 1:size(neighbors, 1)
                        q = neighbors(k, :);
                        if q(1) >= 1 && q(1) <= height && q(2) >= 1 && q(2) <= width && ...
                           BW(q(1), q(2)) == 1 && L(q(1), q(2)) == 0
                            L(q(1), q(2)) = num;
                            queue = [queue; q];
                        end
                    end
                end
                stats(num).PixelList=queue_store;
                stats(num).Area=size(queue_store,1);
            end
        end
    end
end

% Convolution functions
function D_2=doconv(A,B)
ma=size(A,1);   %367
na=size(A,2);   %900
mb=size(B,1);   %3
nb=size(B,2);   %3
% Add 2 rows and 2 columns of borders around the A matrix and set the extra elements to 0
C = padarray(A, [mb-1 nb-1], 0, 'both'); % 371*904
D=zeros(ma+2,na+2);%(369*902)
% Convolution calculation for each position
for j=1:(ma+2)
    for k=1:(na+2)
        D(j,k)=sum(B .* C(j:j+mb-1,k:k+nb-1), 'all');
    end
end
D_2=D(mb:mb+ma-1,nb:na+nb-1);
end
