function xyzStruct = readSceneXYZ(path, whitePoint)

    if nargin < 2
        whitePoint = 'D65';
    end
    
    rgb = im2double(imread(path));
    [h,w,~] = size(rgb);
    
    M = [
        0.4124564 0.3575761 0.1804375;
        0.2126729 0.7151522 0.0721750;
        0.0193339 0.1191920 0.9503041
    ];
    
    XYZ = reshape((M * reshape(rgb,[],3)')', h, w, 3);
    
    xyzStruct.xyz = XYZ;
    xyzStruct.X = XYZ(:,:,1);
    xyzStruct.Y = XYZ(:,:,2);
    xyzStruct.Z = XYZ(:,:,3);
    
    xyzStruct.hfov = 16; % ensures good patch ≈ 65×65
    xyzStruct.whitePoint = whitePoint;

end
