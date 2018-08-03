dirInfo = dir('C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\*.oct');

for k = 1:20

    fprintf('Pullback %d \n', k);
    
    tic
    max_frame = 500;
    
    if k == 10
        max_frame = 375;
    end
    
    if k == 11 
        max_frame = 375;
    end
    
    if k == 12
        max_frame = 375;
    end
    
    for frame = 1:max_frame
        filename = dirInfo(k).name;
        im = imread(['C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\' filename], frame);
        im = log(double(im) + 1.0);
        im = (im - min(im(:)))/(max(im(:)) - min(im(:)));
        im_cart = rectopolar_fast(im2double(im)', 1024);
        %im_cart = repmat(im_cart, 1, 1, 3);
        im_cart = imrotate(im_cart, 90);
        %im_cart = imresize(im_cart, [50 50]);
        imwrite(im_cart, ['E:\OCT_XY\' filename(1:end-4) '_Frame_' int2str(frame) '.tif']);
    end
    
    fprintf('Pullback took %f seconds \n', toc);
    
end