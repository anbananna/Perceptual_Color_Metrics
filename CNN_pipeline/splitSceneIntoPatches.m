function patches = splitSceneIntoPatches(xyz, patchDeg, sceneHFOV, overlap)
% Returns XYZ patches of given size in degrees (with overlap)
    
    [H, W, ~] = size(xyz);
    
    % Convert degrees → pixels
    patchPx = round((patchDeg / sceneHFOV) * W);
    
    % Stride from overlap
    stride = max(1, round(patchPx * (1 - overlap)));
    
    patchCells = {};
    k = 1;
    
    for y = 1:stride:(H - patchPx + 1)
        for x = 1:stride:(W - patchPx + 1)
            patchCells{k} = xyz(y:y+patchPx-1, x:x+patchPx-1, :);
            k = k + 1;
        end
    end
    
    % Convert cell into 4D tensor
    num = numel(patchCells);
    patchH = size(patchCells{1},1);
    patchW = size(patchCells{1},2);
    
    patches = zeros(patchH, patchW, 3, num, 'single');
    for i = 1:num
        patches(:,:,:,i) = patchCells{i};
    end
end
