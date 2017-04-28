
% Matlab_Read_t10k-images_idx3.m
%read t10k-images.idx3-ubyte and convert it into 60000 images
% Written By DXY@HUST IPRAI
% 2009-2-22
clear all;
clc;
[FileName,PathName] = uigetfile('*.*','Please select the file t10k-images.idx3-ubyte');
TrainFile = fullfile(PathName,FileName);
fid = fopen(TrainFile,'r');
a = fread(fid,16,'uint8');
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);
ImageRow = ((a(9)*256+a(10))*256+a(11))*256+a(12);
ImageCol = ((a(13)*256+a(14))*256+a(15))*256+a(16);
if ((MagicNum~=2051)||(ImageNum~=60000))
    error('It is not MNIST t10k-images.idx3-ubyte£¡');
    fclose(fid);    
    return;    
end
savedirectory = uigetdir('','Please select the route of saving£º');
h_w = waitbar(0,'Processing >>');
for i=1:ImageNum
    b = fread(fid,ImageRow*ImageCol,'uint8');   
    c = reshape(b,[ImageRow ImageCol]);
    d = c';
    e = 255-d;
    e = uint8(e);
    savepath = fullfile(savedirectory,[num2str(i) '.bmp']);
    imwrite(e,savepath,'bmp');
    waitbar(i/ImageNum);
end
fclose(fid);
close(h_w);